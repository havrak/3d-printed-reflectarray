"""Add constant columns to CST quoted tab-separated exports.

Usage
-----
Single file, new file next to it (classic behaviour)::

    csv_add_columns.py --file navigator.csv --ratio 1 --P_ratio -0.4

Rewrite in place, original kept as <name>_bak_<timestamp><suffix>::

    csv_add_columns.py --file navigator.csv --replace --ratio 1

Every navigator below a directory tree, in place::

    csv_add_columns.py --dir results/ --replace --ratio 1

Rules
-----
* A column that is already present in the header is NEVER overwritten: the
  requested value is dropped and the original values are kept.  There is no
  override flag, by design.  This is decided per column, so mixing existing and
  new columns in one invocation is fine.
* Files where nothing would be added are left completely untouched (no rewrite,
  no backup), which keeps ``--dir --replace`` runs from churning the tree.
* ``--dir`` with a plain ``updated_`` output pattern would put every result in
  the tree, so it needs ``--replace``; without it the command is refused.
* A row carrying MORE fields than the header has is an error (nothing is
  written): appending a column would silently misalign it.  Short rows just get
  the new trailing constant; blank lines round-trip as blank lines.
* Undo: copy the ``_bak_`` file back, or ``git checkout``.  Auto-restoring a
  backup ONLY happens when the backup carries none of the columns we just added
  but does carry every header of the rewritten file - i.e. the backup is an
  already-annotated sibling rather than the original; that happens when a later
  ``--dir --replace`` pass asking for an extra column sweeps an already
  annotated tree and backs up its own parents.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

# Suffixes considered tabular when walking --dir.
TEXT_SUFFIXES = {".csv", ".txt", ".tsv", ".tab", ".dat"}


def split_args(argv):
    """Separate declared script options from ``--column value`` pairs.

    The column names are not known in advance, so they cannot be declared to
    argparse.  We walk the raw argv instead and hand argparse only the options
    it knows about, preserving their values even when those begin with '-'
    (e.g. ``--P_ratio -0.4``).
    """
    opts, unknown = [], []
    known_with_value = {"--file", "--dir", "--output"}

    i = 0
    while i < len(argv):
        tok = argv[i]
        name, eq, inline_value = tok.partition("=")
        if name in known_with_value and eq:
            opts.append(tok)  # already self-contained: --dir=results
            i += 1
        elif tok in known_with_value:
            if i + 1 >= len(argv):
                sys.exit(f"Error: {tok} requires a value.")
            opts.extend([tok, argv[i + 1]])
            i += 2
        elif tok == "--replace":
            opts.append(tok)
            i += 1
        else:
            unknown.append(tok)
            i += 1
    return opts, unknown


def parse_columns(unknown):
    """Turn leftover tokens into an ordered ``{column: value}`` mapping."""
    new_columns = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok == "--":
            i += 1
            continue
        if tok.startswith("--"):
            name, eq, inline_value = tok.partition("=")
            col_name = name.lstrip("-")
            if not col_name:
                sys.exit(f"Error: malformed column flag {tok!r}.")
            if col_name in new_columns:
                sys.exit(f"Error: column {col_name!r} given more than once.")
            if eq:
                new_columns[col_name] = inline_value
                i += 1
            # A following token that is not itself a "--flag" is the value.
            elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                new_columns[col_name] = unknown[i + 1]
                i += 2
            else:
                new_columns[col_name] = ""
                i += 1
        else:
            sys.exit(f"Error: unexpected argument {tok!r} (columns must be given as --name value).")
    return new_columns


def read_table(path):
    """Read a quoted tab-separated file into (header, rows).

    Blank lines are kept as empty lists so that a rewrite round-trips them.
    Short rows are tolerated; rows with more fields than the header are rejected
    in read_table_mapped().  Raises ValueError without a usable header.
    """
    csv.field_size_limit(min(sys.maxsize, 10**9))
    with open(path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.reader(f_in, delimiter="\t", quotechar='"')
        rows = [[field.strip() for field in row] for row in reader]
    if not rows or not any(field.strip() for field in rows[0]):
        raise ValueError("file is empty or has no header")
    return rows[0], rows[1:]


def read_table_mapped(path):
    """Read a table, mapping each field to its header column.

    Returns ``(headers, mapped_rows)`` where each mapped row is a dict keyed by
    header name, or ``None`` for a blank line.  Raises ValueError if the file has
    no usable header or a row carries more fields than the header.
    """
    headers, raw_rows = read_table(path)
    if len(set(headers)) != len(headers):
        dupes = sorted({h for h in headers if headers.count(h) > 1})
        raise ValueError(f"duplicate header name(s) {dupes}")
    mapped = []
    for index, row in enumerate(raw_rows, start=2):
        if not row:
            mapped.append(None)
            continue
        if len(row) > len(headers):
            raise ValueError(
                f"line {index} has {len(row)} fields but the header has {len(headers)}"
            )
        mapped.append(dict(zip(headers, row)))
    return headers, mapped


def render_table(headers, rows, extras):
    """Serialise header + rows with ``extras`` appended, CST style (all quoted)."""
    out = ['\t'.join(f'"{h}"' for h in headers + list(extras.keys()))]
    for row in rows:
        if row is None:
            out.append("")
        else:
            out.append('\t'.join(f'"{v}"' for v in row + list(extras.values())))
    return "\n".join(out) + "\n"


def backup_path(path):
    """<stem>_bak_<YYYYmmdd-HHMMSS><suffix>, de-duplicated if called twice in a second."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}_bak_{stamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_bak_{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


def process_file(path, new_columns, args):
    """Apply the annotation to one file. Returns (status, message, columns_added)."""

    def finish(status, msg, added, bak=None):
        # Remember what we did so a later pass can vet the backup ourselves.
        if args.replace and status in ("updated", "unchanged"):
            args.log.append({"path": path, "status": status,
                             "added": added, "bak": bak})
        return status, msg, added

    bak = None
    try:
        headers, mapped_rows = read_table_mapped(path)
    except (OSError, UnicodeDecodeError, ValueError, csv.Error) as exc:
        return finish("error", f"could not read: {exc}", [])

    # Per-column rule: existing header wins, always.
    columns_to_add = {k: v for k, v in new_columns.items() if k not in headers}
    skipped = [k for k in new_columns if k in headers]

    if not columns_to_add:
        return finish("unchanged", f"all requested columns already present (kept: {skipped})", [])

    # A row that is short of trailing fields is fine: the requested columns are
    # the trailing ones, so a short row simply had no value there.
    rows = [[] if r is None else [r.get(h, "") for h in headers] for r in mapped_rows]

    text = render_table(headers, rows, columns_to_add)

    if args.replace:
        out_path = path
        # Always rewrite under --replace: a fresh backup is the user's undo, so
        # the file must stay in the freshly serialised canonical form.
        bak = backup_path(path)
        path.replace(bak)
        try:
            with open(out_path, "w", encoding="utf-8", newline="") as f_out:
                f_out.write(text)
        except OSError:
            bak.replace(path)  # restore, never leave the original missing
            raise
        msg = f"rewritten in place, backup -> {bak.name}"
    else:
        out_path = Path(args.output) if (args.output and not args.directory) else \
            path.with_name(f"updated_{path.name}")
        with open(out_path, "w", encoding="utf-8", newline="") as f_out:
            f_out.write(text)
        msg = f"written to {out_path}"

    if skipped:
        msg += f"; existing columns left untouched: {skipped}"
    return finish("updated", msg, list(columns_to_add), bak)


def candidate_files(directory):
    """All plain files under ``directory`` that look tabular and are not backups."""
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or "_bak_" in p.name:
            continue
        if p.suffix.lower() in TEXT_SUFFIXES:
            yield p


def vet_backups(log):
    """Undo a ``--dir --replace`` pass that back-filled a whole tree.

    A backup is a *victim* of a directory pass -- not the original of the file we
    just changed -- when its header is disjoint from the columns we added AND it
    contains every header of the new file.  In that case the backup is the
    inverse annotation of the same file, so restore the original from it.
    """
    restored = 0
    for entry in log:
        bak = entry["bak"]
        if entry["status"] != "updated" or bak is None:
            continue
        try:
            bak_headers, _ = read_table(bak)
            new_headers, _ = read_table(entry["path"])
        except (OSError, UnicodeDecodeError, ValueError, csv.Error):
            continue
        if set(bak_headers) & set(entry["added"]):
            continue  # backup predates our annotation: a legitimate original
        if not set(new_headers) <= set(bak_headers):
            continue  # shapes we do not understand: leave the backup alone
        bak.replace(entry["path"])
        entry["undone"] = True
        print(f"RESTORED {entry['path']}: dropped {entry['added']}, "
              f"{bak.name} was a parent (already-annotated) file")
        restored += 1
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Add constant columns to CST quoted tab-separated exports.",
        epilog="Column values are given as free-form --name value pairs; columns "
               "that already exist in the header are never overwritten.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="single input file to annotate")
    group.add_argument("--dir", help="directory searched recursively for tabular files")
    parser.add_argument("--replace", action="store_true",
                        help="rewrite the file in place, keeping a "
                             "<name>_bak_<timestamp> copy of the original")
    parser.add_argument("-o", "--output", default=None,
                        help="output path, only valid together with --file and "
                             "without --replace (default: updated_<name>)")

    # The free-form column pairs must be consumed before anything else, so a
    # value like -0.4 is not mistaken for an option.
    opts, leftover = split_args(sys.argv[1:] if argv is None else argv)
    args, unknown = parser.parse_known_args(opts)
    args.directory = args.dir is not None
    args.log = []

    new_columns = parse_columns(unknown + leftover)
    if not new_columns:
        print("No constants to add. Specify them using the format: --ratio 1 --P_ratio -0.4")
        return 0

    if args.output and (args.replace or args.dir):
        parser.error("--output cannot be combined with --replace or --dir.")

    # --dir needs --replace: without it every result would land next to its input
    # as updated_<name>, which is not a useful bulk behaviour.
    if args.dir and not args.replace:
        parser.error("--dir requires --replace (in-place annotation of the tree).")

    if args.file:
        target = Path(args.file)
        if not target.is_file():
            print(f"Error: {target} is not a file.")
            return 1
        targets = [target]
    else:
        root = Path(args.dir)
        if not root.is_dir():
            print(f"Error: {root} is not a directory.")
            return 1
        targets = list(candidate_files(root))
        if not targets:
            print(f"No tabular files found under {root}.")
            return 0

    counts = {"updated": 0, "unchanged": 0, "error": 0}
    # Collect the results before printing: vetting backups can undo some of the
    # rewrites that were just reported as updated.
    results = []
    for path in targets:
        try:
            results.append((path,) + process_file(path, new_columns, args))
        except OSError as exc:
            results.append((path, "error", f"filesystem error: {exc}", []))

    if args.replace:
        vet_backups(args.log)
    undone = {e["path"]: e["added"] for e in args.log if e.get("undone")}

    for path, status, msg, added in results:
        if path in undone:
            status, msg = "unchanged", f"already annotated with {undone[path]}, restored"
        counts[status] += 1
        if status == "updated":
            print(f"UPDATED  {path}: added {added}; {msg}")
        elif status == "unchanged":
            print(f"SKIPPED  {path}: {msg}")
        else:
            print(f"ERROR    {path}: {msg}")

    print(
        f"\n{counts['updated']} updated, {counts['unchanged']} skipped, "
        f"{counts['error']} error(s) out of {len(targets)} file(s)."
    )
    return 1 if counts["error"] else 0


if __name__ == '__main__':
    sys.exit(main())
