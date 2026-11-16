"""
Shared data loading and geometry logic for the data-statistics scripts.

Extracted from stats.py so that the table renderer and the coverage statistics
read the tree the same way: same CSV parser, same parameter grouping, same
clearance rule. Nothing here draws anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #

def _split_strip(line: str) -> list[str]:
    """TAB-split, then strip quotes and whitespace from each field."""
    return [part.strip().strip('"').strip() for part in line.split("\t")]


def _to_float(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def read_navigator(path: Path) -> list[dict[str, float]]:
    """
    Read one navigator CSV into a list of parameter dicts.

    CST writes TAB-separated fields, each wrapped in double quotes, with a
    leading space inside the quotes on the header. Column names vary between
    export scripts, so lookup is by name rather than position.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        print(f"  cannot read {path}: {exc}")
        return []

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    header = _split_strip(lines[0])
    rows: list[dict[str, float]] = []

    for line in lines[1:]:
        fields = _split_strip(line)
        if len(fields) < len(header):
            fields += [""] * (len(header) - len(fields))

        record: dict[str, float] = {}
        for name, field in zip(header, fields):
            value = _to_float(field)
            if value is not None:
                record[name] = value
        if record:
            rows.append(record)

    return rows


# --------------------------------------------------------------------------- #
# Blacklist matching
# --------------------------------------------------------------------------- #

# The wildcard token: a parameter mapped to this matches any value.
WILDCARD = "*"

# Float comparison tolerance. Values come straight from the CST sweep table, so
# exact equality would usually suffice; the tolerance only guards against a
# value being written with a different number of decimals in different exports.
BLACKLIST_TOL = 1e-6


def matches_blacklist(
    params: dict[str, float],
    blacklist: list[dict[str, float | str]],
) -> bool:
    """
    True when `params` satisfies any complete entry of `blacklist`.

    Each entry is a dict of raw parameter name to required value. A parameter
    absent from an entry, or mapped to `WILDCARD`, matches anything. All
    conditions within one entry must hold for that entry to match, and a match
    on any entry excludes the sample.

    Parameter names are compared case-insensitively and with surrounding
    whitespace ignored, matching the way the CSV reader resolves columns.

    This mirrors `metadata.matches_blacklist` in the data-processor directory.
    The logic is duplicated deliberately so the two tools stay independent; the
    blacklists themselves must be kept in step by hand.

    Parameters
    ----------
    params    : raw parameter values for one configuration, e.g. one row from
                `read_navigator`
    blacklist : list of condition dicts, from config.BLACKLIST
    """
    if not blacklist:
        return False

    # Normalise the row keys once, so lookup does not depend on the export's
    # capitalisation.
    normalised = {k.strip().lower(): v for k, v in params.items()}

    for entry in blacklist:
        matched = True
        for key, want in entry.items():
            if want == WILDCARD:
                continue

            have = normalised.get(key.strip().lower())
            if have is None:
                matched = False
                break

            try:
                want_f = float(want)
            except (TypeError, ValueError):
                raise ValueError(
                    f"blacklist entry {entry!r}: value for {key!r} is neither "
                    f"a number nor {WILDCARD!r}"
                )

            if abs(float(have) - want_f) > BLACKLIST_TOL:
                matched = False
                break

        if matched:
            return True

    return False


def describe_blacklist_entry(entry: dict[str, float | str]) -> str:
    """Human-readable form of one blacklist entry, for logging."""
    if not entry:
        return "(empty entry: matches everything)"
    return ", ".join(f"{k}={v}" for k, v in entry.items())


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def max_major_axis(
    p_ratio: float,
    ratio: float,
    clearance_mm: float = 0.0,
) -> float:
    """
    Largest semi-major axis whose bounding box still fits inside the cell.

    `clearance_mm` is the gap left between the ellipse bounding box and the cell
    wall. Pass 0.0 for the strict geometric limit, or the validity clearance to
    reproduce the filtering rule used elsewhere.
    """
    cell_mm = p_ratio * config.LAMBDA0_MM
    half = cell_mm / 2.0 - clearance_mm
    if half <= 0 or ratio <= 0:
        return float("nan")
    return half * (2.0 ** 0.5) / (1.0 + 1.0 / ratio ** 2) ** 0.5


def fits_with_clearance(
    a_mm: float,
    p_ratio: float,
    ratio: float,
    clearance_mm: float = config.VALIDITY_CLEARANCE_MM,
) -> bool:
    """True when the ellipse bounding box leaves at least the clearance."""
    limit = max_major_axis(p_ratio, ratio, clearance_mm)
    if not (limit == limit):        # NaN
        return False
    return a_mm <= limit


def extend_sweep(
    a_max: float,
    p_ratio: float,
    ratio: float,
    step_mm: float = config.TARGET_STEP_MM,
    margin_mm: float = config.EDGE_MARGIN_MM,
) -> tuple[float, float, int] | None:
    """
    Work out the extra samples needed to reach the geometric edge.

    Returns (start, end, n) or None when the sweep already reaches the edge.
    The new interval begins one step after the last sampled point and ends as
    close to the limit as a step of at most `step_mm` allows.
    """
    limit = max_major_axis(p_ratio, ratio) - margin_mm
    if not (limit == limit) or a_max >= limit:
        return None

    start = a_max + step_mm
    if start > limit:
        return None

    n_new = int((limit - a_max) / step_mm)
    if n_new < 1:
        return None

    end = a_max + n_new * step_mm
    if limit - end <= step_mm:
        end = limit
    return start, end, n_new


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class Sample:
    """One simulated configuration, kept at its original granularity."""

    p_ratio: float
    epsilon_r: float
    h_pillar: float
    ratio: float
    a: float
    source: str

    @property
    def levels(self) -> tuple[float, ...]:
        return (self.p_ratio, self.epsilon_r, self.h_pillar, self.ratio)

    @property
    def a_limit(self) -> float:
        """Largest usable semi-major axis, zero clearance."""
        return max_major_axis(self.p_ratio, self.ratio)

    @property
    def is_valid(self) -> bool:
        """True when the sample passes the validity clearance."""
        return fits_with_clearance(self.a, self.p_ratio, self.ratio)


@dataclass
class Leaf:
    """A `ratio`-level group: one geometry, summarised over its `a` samples."""

    levels: tuple[float, ...]
    n_samples: int
    a_min: float
    a_max: float
    source: str

    @property
    def a_span(self) -> float:
        return self.a_max - self.a_min

    @property
    def p_ratio(self) -> float:
        return self.levels[0]

    @property
    def epsilon_r(self) -> float:
        return self.levels[1]

    @property
    def h_pillar(self) -> float:
        return self.levels[2]

    @property
    def ratio(self) -> float:
        return self.levels[3]

    @property
    def a_limit(self) -> float:
        return max_major_axis(self.p_ratio, self.ratio)

    @property
    def extension(self) -> tuple[float, float, int] | None:
        return extend_sweep(self.a_max, self.p_ratio, self.ratio)

    @property
    def needs_extension(self) -> bool:
        return self.extension is not None

    @property
    def a_coverage(self) -> float:
        limit = self.a_limit - config.EDGE_MARGIN_MM
        if not (limit == limit) or limit <= 0:
            return float("nan")
        return self.a_max / limit


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def collect_samples(root: Path, verbose: bool = False) -> tuple[list[Sample], list[str]]:
    """
    Walk the tree and read every navigator CSV into individual samples.

    Column names are matched case-insensitively, and rows missing a required
    value are skipped with a count, because a column present in the header does
    not guarantee it is present in every row.
    """
    samples: list[Sample] = []
    warnings: list[str] = []

    nav_files = sorted(root.rglob(config.NAVIGATOR_NAME))
    if not nav_files:
        return [], [f"no {config.NAVIGATOR_NAME} found under {root}"]

    required = config.LEVELS + [config.TUNING]

    for nav in nav_files:
        rel = nav.parent.relative_to(root)
        rows = read_navigator(nav)
        if not rows:
            warnings.append(f"{nav}: no usable rows")
            continue

        if verbose:
            print(f"  {rel}: {len(rows)} rows, header {list(rows[0])}")

        available = {name.strip().lower(): name for name in rows[0]}
        resolved: dict[str, str] = {}
        for want in required:
            hit = available.get(want.strip().lower())
            if hit is None:
                resolved = {}
                break
            resolved[want] = hit

        if not resolved:
            missing = [w for w in required if w.lower() not in available]
            warnings.append(
                f"{nav}: missing columns {missing}, skipped"
                f"  (header was {list(rows[0])})"
            )
            continue

        skipped = 0
        blacklisted = 0
        for row in rows:
            if any(resolved[k] not in row for k in required):
                skipped += 1
                continue

            # Blacklist check on the raw CSV values, before the sample is built.
            # Uses the row's own keys, so the case-insensitive column
            # resolution above does not affect the lookup.
            if matches_blacklist(row, config.BLACKLIST):
                blacklisted += 1
                continue

            samples.append(Sample(
                p_ratio=row[resolved["P_ratio"]],
                epsilon_r=row[resolved["epsilon_r"]],
                h_pillar=row[resolved["H_pillar"]],
                ratio=row[resolved["ratio"]],
                a=row[resolved["a"]],
                source=str(rel),
            ))

        if blacklisted:
            warnings.append(
                f"{rel}: {blacklisted} row(s) excluded by the blacklist"
            )

        if skipped:
            warnings.append(
                f"{nav}: {skipped} of {len(rows)} rows lacked a required value"
            )

    return samples, warnings


def collect_leaves(
    root: Path,
    verbose: bool = False,
    drop_invalid: bool = False,
) -> tuple[list[Leaf], list[str]]:
    """
    Group samples by their four level values and summarise each group's `a`
    coverage.

    With `drop_invalid` the samples failing the validity clearance are removed
    before grouping, so the reported ranges describe only physically valid
    geometry.
    """
    samples, warnings = collect_samples(root, verbose=verbose)

    if drop_invalid:
        kept = [s for s in samples if s.is_valid]
        dropped = len(samples) - len(kept)
        if dropped:
            warnings.append(
                f"{dropped} of {len(samples)} samples removed by the "
                f"{config.VALIDITY_CLEARANCE_MM} mm validity clearance"
            )
        samples = kept

    groups: dict[tuple[float, ...], dict] = {}
    for s in samples:
        key = tuple(
            round(v / config.VALUE_TOL) * config.VALUE_TOL for v in s.levels
        )
        entry = groups.setdefault(key, {"a": [], "source": s.source})
        entry["a"].append(s.a)

    leaves: list[Leaf] = []
    for key, entry in groups.items():
        a_values = entry["a"]
        leaves.append(Leaf(
            levels=key,
            n_samples=len(a_values),
            a_min=min(a_values),
            a_max=max(a_values),
            source=entry["source"],
        ))

    leaves.sort(key=lambda lf: lf.levels)
    return leaves, warnings


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def fmt(value: float) -> str:
    """Compact number formatting: integers lose the decimal point."""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:g}"


def resolve_root(given: Path | None) -> Path:
    """
    Resolve the data root.

    A relative path is tried against the current directory first, then against
    the repository root (four levels above this file), so the scripts work from
    anywhere inside the repo.
    """
    if given is None:
        given = config.DEFAULT_DATA_ROOT
    if given.is_absolute():
        return given
    if given.exists():
        return given.resolve()
    return (Path(__file__).resolve().parents[4] / given).resolve()
