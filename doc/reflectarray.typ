#import "config.typ": *

#align(center)[
    #text(size: 18pt, weight: "bold")[Reflectarray with LP $->$ CP Polarization Conversion] \
    #text(size: 14pt)[Deepdive on potential thesis topic] \
    #link("https://typst.app/project/wBQGVX8CTMedKqXLVoBbtb")[Online Version]
]


#show: conf

#outline()

#pagebreak()
= Reflectarray with LP $->$ CP Polarization Conversion
This document focuses on a potential topics of reflectarrays with unit cells that exhibit form-birefringerence -- that is they exhibit anisotropic properties. It builds on the general reflectarray and material observations recorded in `reserach.typ` file.

#pagebreak()
= The Idea: Form-Birefringent Unit Cells

== Physical Mechanism
- Form birefringence creates an anisotropic medium using subwavelength features, so the incident wave does not resolve the individual physical boundaries and instead sees a homogenized anisotropic medium.
  - This can be achieved either with asymmetric voids/pillars, or by combining two or more isotropic materials into a subwavelength grating whose composite response is anisotropic @Isakov2018.
  - Effective refractive index differs for electric fields polarized parallel ($n_parallel$) versus perpendicular ($n_perp$) to the structural axis.
- *Design rule for LP-to-CP conversion:*
  - When a linearly polarized wave is incident at $45 degree$ to the structural axis, its parallel and perpendicular components travel at different phase velocities and emerge with a differential phase delays.
    $ Delta phi = (2 pi f d) / c_0 (n_parallel - n_perp) $
  - Setting $Delta phi = pi / 2$ (quarter-wave retardation) while keeping the two transmitted amplitudes roughly equal (This won't be possible to achieve.) converts the linear input into a circularly polarized output.
  - This was actually used by @Hao2020 where the unit cell had this anisotropy, however phase shifts was mainly a function of height

== Reflectarray Application
- Reflectarray can be designed with unit cell that exhibits such anisotropy to have additional function aside from beam focusing
  - *Polarization transformation* (CP generation or LP-to-CP conversion).
  - *Polarization purification:* since most real feeders exhibit some cross-polar reception, the array can be designed so that only the desired polarization is focused, thereby improving cross-polar discrimination (XPD) of the overall system.
- Basic ideal for the new cell design comes from @Massaccesi2023
  - @Massaccesi2023 achieved a flat reflectarray, comprised of unit cell with square footprint with variable square hole.
  - New cell design would have an elliptical (or rhombic) cutout in place of a symmetric void - with major axis of the body being oriented along the diagonal of the square.
  - the ellipse/rhombus presents a different effective permittivity along its major and minor axes, and thus a different reflection phase per polarization.
  - Conventional polarization-controlling reflectarrays rely on blocks of differing height to obtain the per-axis phase, like @Cheng2022
  - What hasn't really been tried is constructing an array of uniform height where the two degrees of freedom (necessary to tune $n_perp$ and $n_parallel$ independently) are achieved in geometry changes in $x$ and $y$ directions
  - Such design could enable easier addition of some covering to make the final reflectarray smooth at the surface futher enhancing it's mechanical properties
  - Or possibly in order to improve printability the unit cell could be inverted -- instead of elliptical cutouts printing elliptical pillars.
- Phase shifting mechanism of the cell will then rely on it operating near resonance
  - Single resonance can give up to 360 degrees of phase change.
  - Due to Form Form-Birefringence the resonance will be slightly shifted between two axis.

== Geometry

=== Array Size

Using basic gain formula
$
  G = eta (4 pi A)/(lambda_0^2),
$

a gain of the reflect array can be estimated. Assuming aperture efficiency of 40~%, closed to what was achieved by @Massaccesi2023 using $0.3 lambda_0$ size cells, @Beccaria2025 achieved 50~% with $0.4 lambda_0$ size. Table bellow shows rough gain estimation for a square aperture (Since the print bed is square there's no reason to consider any other geometry).

#align(center)[
  #table(
    columns: 7,
    align: center,
    [*Aperture Size* ($lambda$)],
    [*Gain* (dBi)],
    [*Size 10 GHz* (mm)],
    [*Size 15 GHz* (mm)],
    [*Size 20 GHz* (mm)],
    [*Size 25 GHz* (mm)],
    [*Size 30 GHz* (mm)],

    [6], [22.5], [180], [120], [90], [72], [60],
    [8], [25.07], [240], [160], [120], [96], [80],
    [10], [27.01], [300], [200], [150], [120], [100],
    [12], [28.60], [360], [240], [180], [144], [120],
    [14], [29.94], [420], [280], [210], [168], [140],
    [16], [31.10], [480], [320], [240], [192], [160],
    [18], [32.12], [540], [360], [270], [216], [180],
    [20], [33.03], [600], [400], [300], [240], [200],
  )
]

=== Unit Cell Size

All text bellow is taken from @Nayeri2018.
The standard rule for reflectarrays is that the unit cell periodicity should be $<= lambda_0/2$ at the design frequency to avoid grating lobes.
Dimension of $lambda_0/2$ seems to be the standard in other literature.

#align(center)[
  #table(
    columns: 3,
    align: center,
    [*Periodicity* ], [*Advantages*], [*Disadvantages*],
    [$lambda_0/2$], [Standard; sufficient phase range ($300 degree$); good for large apertures], [narrower bandwidth],
    [$lambda_0/3$],
    [Better aperture sampling; wider bandwidth; lower element losses],
    [Reduced phase range; steeper phase slope (more quantization error)],

    [$lambda_0/4$],
    [Even wider bandwidth; further loss reduction; better for small-to-moderate arrays],
    [Phase range severely restricted; would necessitate multi-layer design or high $epsilon_r$ material],
  )
]

Especially if the thesis will be focused on creating reflectarray with additional capabilities like polarization transformation it's not worth to consider other cell designs then with $lambda_0/2$ base dimension.

In terms the heigh, higher cell will mean A) larger losses in the dielectric and B) being more susceptible to incidence angle changes.

=== Conclusions for this topic.
Typical FDM printer print bed is in the range of 200 to 300~mm. For SLA most are between 120 to 150~mm.
As the thesis will be mainly research focused a aperture size could be on the lower end - between 8 to 12 $lambda_0$ giving size that should be printable on FDM even at 10 GHz and SLA till 25 GHz assuming 150~mm print bed.

As for unit cell size, bandwidth here isn't the primary consideration.
Usable range of frequencies will be already limited plenty by the range in which polarization conversion will work as intended.
Thus unit cell smaller than $lambda_0/2$ doesn't really make that much sense.



== Problems and Open Questions
- *Phase-range coupling is the central difficulty:*
  - The unit cell must supply $Delta phi_x = 360 degree$ while maintaining $Delta phi_y - Delta phi_x approx 90 degree$ across the entire phase range.
  - This effectively requires a single cell geometry that has to cover range of $0 degree$ to $450 degree$ while the orthogonal axis tracks it with a constant offset.
  - This is a substantially harder problem than a single-polarization element.
  - Block-based CP transformers side-step this by offsetting the $y$ component in height relative to $x$.
- *The phase range of the available reference designs is insufficient:*
  - Assuming a smaller, compact aperture fed by a reasonable illuminator, a reduced maximum phase of roughly $300 degree$ might be tolerable -- but even that demands coverage out to about $390 degree$ in the converted axis.
  - Existing dielectric reflectarray elements fall short of this.
    - @Zhang2017 reached only $270 degree$ with $epsilon_r = 2.75$ and needed a custom composite at $epsilon_r = 4.4$ to cover desired range.
    - @Beccaria2025 required $epsilon_r = 7.5$ to reach roughly $300 degree$.
    - The core reference @Massaccesi2023 covered only about $200 degree$ (The exact value is not stated in the thesis, I'm judging from the phase curves they've listed), though this is quite extreme case.
  - The gap between required and available phase range is therefore the primary feasibility risk, and it pushes the design toward utilizing materials with the highest workable permittivity.
- *Manufacturing feasibility is in tension with the required geometry:*
  - *SLA:* has the resolution and build area to realize a compact reflectarray at higher frequencies, but high-permittivity resins are prohibitively expensive.
  - *FDM:* the filament availability problem is largely solved, but print resolution is the limiting factor -- the unit cells are small and the anisotropic cutout geometry is not straightforward to print.
    - While precision of nozzle positioning on more advance printers is in the ranges of single, tens micrometers nozzle diameter is normally 0.4~mm.
    - While 0.2~mm nozzles are available and I think that Zetamix filaments should still be printable with them, even at 0.2~mm the resolutions is far behind what SLA can achieve.
    - The narrow gaps at the tips of an elliptical cutout may fall below the printer's reliable feature size, causing inconsistent extrusion or incomplete fill; a rhombic cutout fares somewhat better but is not a complete solution.
    - In addition the addition of material on walls -- print being slightly bigger then designed requires consideration during design of the print model - which will slightly differ from the optimized model.
  - Mitigations:
    - Lower the design frequency so that resolution is less critical, at the cost of build-area problems.
    - Invert the design so that the anisotropic features are pillars rather than voids, avoiding thin walls that must be dimensionally accurate.
- *Design challenges:*
  - Because the two polarizations are coupled through a single geometry, the per-axis reflection phases cannot be tuned independently.
    - Incidence angle stability is up to questions. I would assume that ellipse would fair better then rhombus, but how well will the phase delay properties be kept across the array is hard to estimate.
  - Computer optimization is required to converge on a cell that meets both the total range and the $90 degree$ offset simultaneously.
  - I don't think this will be that big of a problem - S parameters for many different configuration of major - minor axis combos can be computed, surrogate model with Gaussian Processes or Polynomial Response created and configurations for each phase delay discovered on that.

#pagebreak()
= Materials
- The unit cell is an all-dielectric, additively manufactured, so the material system sets both the achievable permittivity contrast (and therefore the attainable phase range) and the loss.
- *FDM with Zetamix* $"TiO"_2$ (up to $epsilon_r = 10$) is the leading candidate: uniform filler distribution, a reproducible permittivity ladder, and no sedimentation issue.
- *SLA with a manually loaded resin* offers resolution and build-area headroom but carries the sedimentation risk and an unverified printability assumption.


== The Baseline-Polymer Permittivity Ceiling
- Unloaded SLA and FDM polymers usually exhibit low relative permittivity, roughly $epsilon_r$ between 2.0 and 3.0.
  - This ceiling limits the phase range obtainable from geometry alone, which is exactly the constraint identified above.
  - A polarization-converting cell needs independent control of two orthogonal reflection phases, so the phase-range limitation is even more restrictive than for a single-polarization reflector.
- Micro-voids from high feedstock viscosity and the natural result of FDM layering process depress the bulk dielectric constant.
- Hygroscopic polymers take up moisture, which further raises $"tan" delta$ and lowers $epsilon_r$.

== High-Permittivity FDM Filaments
- High-permittivity filaments are made by blending the polymer matrix with ceramic nanoparticles such as barium titanate, titanium dioxide, or alumina.
  - For FDM systems the ceramic filler does not sediment as it does in liquid photopolymer systems, so the filler distribution remains uniform across the print.
- @ZetamixEpsilon offers a family of $"TiO"_2$-loaded materials with $epsilon_r$ of 2.2, 4.5, 7.5, and 10 at 100~% infill.
  - At 35~% infill the nominal $epsilon_r = 7.5$ material drops to around $epsilon_r = 2$.
- @Zhang2017 obtained comparable material by manually adding submicron additives to a base PREPERM filament.
- Practical conclusion for this thesis:
  - The Zetamix $"TiO"_2$ family is the most reproducible FDM option.
  - It's also easiest to get filament with high permittivity for any plausible 3D printer type.
  - The $epsilon_r = 10$ grade provides headroom to test whether the required phase shifts are even realizable before committing to a lower-permittivity material.

== Photopolymer and Ceramic-Loaded Resins (SLA)
- In SLA and other liquid-resin processes, dense ceramic fillers undergo gravity-driven sedimentation before UV curing.
  - This induces an unintended anisotropic permittivity gradient along the vertical build axis.
  - Non-uniform particulate distribution also inhibits the maximum achievable effective permittivity.
- Most ceramic-loaded photocurable resins do not reach high permittivity because the additive concentration is kept small
  - @Whittaker2023 indicates values around $epsilon_r approx 4$.
  - This is below what the phase-range analysis demands.
- Higher permittivity resins are created by loading base resin with $"BaTiO"_3$ nanoparticles.
  - Manufacturer data for 3Dresyn HP1 BTO70 lists $epsilon_r$ between 8 and 9 with loss tangent below 0.04 across 10-20~GHz range @3DresynHP1BTO70.
  - However, they are prohibitively expansive.
  - Loss factor of these resins is usually rather high -- base resin is already fairly lossy, and barium titanate suffers from additional dipole relaxation losses on higher frequencies.

#pagebreak()
= Measurement of the Unit Cell #TODO
- I need to measure phase delay in two principal axis at 90 degree incident angle illuminated by a lineary polarized source.
  - Waveguide out of question -- incidence angle changes with frequency.
  - Dielectric waveguide esque structure could maybe be used.
  - Free space to complicated.
  - Parallel plate?

#pagebreak()
= Simulation of Unit cell
- An all-dielectric unit cell with:
  - A square footprint of $lambda_0 / 2 times lambda_0 / 2$ where $P = c_0 / (2 f_0)$.
  - An elliptical pillar whose major axis is aligned with the *diagonal* of the square, i.e. rotated $45 degree$ relative to the array axes.
  - A base substrate layer below the pillar.
  - $epsilon_r in [7, 10]$; the headroom value used in the study is $epsilon_r = 9$.
- Because the ellipse presents a different projected cross-section to the x- and y-polarized waves, the two polarizations see different effective permittivities. This is the form-birefringence mechanism.

== CST Setup

=== Create the project
- `File -> New -> Microwave & RF / Optical -> Unit Cell`.
- Set the frequency range (e.g. 25-35 GHz for Ka-band) and units (mm, GHz, ns).

=== Define parameters
- Open `Home -> Parameter List` (`Ctrl + P`) and add:

#align(center)[
  #table(
    columns: 3,
    align: center,
    [*Name*], [*Value*], [*Description*],
    [`f0`], [30], [Design frequency (GHz)],
    [`P`], [5.0], [Unit cell period, $lambda_0 / 2$ at 30 GHz],
    [`epsilon_r`], [9], [Material permittivity],
    [`a`], [1.5], [Ellipse semi-major axis (mm)],
    [`ratio`], [2], [Aspect ratio $a / b$, always $> 1$],
    [`b`], [`a / ratio`], [Ellipse semi-minor axis (mm)],
    [`H_pillar`], [3.8], [Pillar height (mm)],
    [`H_base`], [0.6], [Base substrate thickness (mm)],
  )
]

- Defining `b = a / ratio` with `ratio > 1` enforces $a > b$ automatically, so the sweep only searches half of the parameter space.

=== Build the geometry
- *Ground plane:* `Modeling -> Brick` (`Ctrl + B`), name `ground`, X and Y from `-P/2` to `P/2`, Z from `-H_base` to `0`, material `PEC`.
- *Base dielectric:* `Ctrl + B`, name `base_dielectric`, X and Y from `-P/2` to `P/2`, Z from `0` to `H_base`, new material `Zetamix_eps9` with Epsilon set to `epsilon_r`.
- *Elliptical pillar:* CST has no native elliptical-cylinder primitive, so extrude a 2-D ellipse:
  - `Curves -> Ellipse`: centre `(0, 0, 0)`, radius X `= a`, radius Y `= b`.
  - `Curves -> Transform -> Rotate`: select the curve, axis z, angle `45`.
  - `Modeling -> Extrude Curves`: select the rotated ellipse, height `H_pillar`, name `pillar`.
  - `Modeling -> Transform -> Translate`: `pillar`, Z `= H_base`.
- *Boolean union:* `Modeling -> Boolean -> Add`, first `base_dielectric`, then `pillar`.

=== Boundaries and ports
- The Unit Cell template pre-configures these; verify them:
  - `Simulation -> Boundaries`: X and Y set to `unit cell`, Zmin set to `electric (Et = 0)`, Zmax set to `open (add space)`.
  - `Simulation -> Ports`: the Floquet port exists by default. Open its properties and ensure Mode 1 (TE00) and Mode 2 (TM00) are enabled.
- #INFO because the pillar sits at $45 degree$, the structural eigenmodes are rotated relative to the global axes. Cross-terms `S1,2` and `S2,1` will therefore be non-zero. This is an expected consequence of the birefringence, not an error.

=== Solvers
- Frequency domain simulator was used.
- Adaptive meshing was used, to secure it's accuracy while maintaining good perforamce a frequency bandwidth od 1 GHz at $f_0=30$~GHz was used.

== Simulations Results
- Initial simulations ran with low sampling of `a` and with `ratio=1`.
  - Simulations results were similar to those seen in @Beccaria2025.
  - Structure behaved generally as expected - at higher permittivities ($>= 7$) structure exhibited multiple resonance at reasonable height, leading to the slope being too steep.
  - Within a single resonance the highest achieved phase range with slope of $<= 400 degree/"mm"$  was around $320 degree$.
  - Void structure didn't exhibit significantly better behavior in order to justify it.
  - Later when `ratio` was introduced phase range of the cell decreased significantly.
- Behavior of the UC was too complex to optimize by hand $->$ surrogate model based modeling and optimization on that model was used.
- #TODO Finish this sections.
  - Conclusion of the preliminary study.
  - Need to export plots from the data explorer, not just for `ratio=1`.
  - Idk more boring stuff.



#pagebreak()
= Surrogate Model

== Why a Surrogate Is Needed
- The feasibility study established that the phase behavior of the elliptical pillar cannot be predicted analytically.
  - There is no closed-form solution for a truncated elliptical dielectric resonator on a grounded substrate
- At the same time the design problem is not a single evaluation but a search.
  - Each element of the final reflectarray needs a geometry that realises its own required phase, and the array contains hundreds of elements.
  - On top of that the geometry has to be chosen such that the $90 degree$ inter-axis offset is preserved across the whole phase range.
- Evaluating that by full-wave simulation is not viable:
  - A single CST solve costs tens of seconds if not minutes.
  - The design space spans five parameters, and the search would have to be repeated for every element of every candidate array.
  - CST Doesn't produce results in eigenmodes - so before any optimizer would run results would need to be rotated.
- A surrogate model replaces the full-wave solve with a cheap regression that maps geometry to response,
  - Search runs on the model and only the final candidates are verified in CST.
  - This is the standard role of surrogate modelling in simulation-driven design @Forrester2008.
  - TODO: later go through Forrester2008 bit more and make the text more complete. It doesn't talk about MLP modeling, ends on support vectors predictors, who precede MLPs.

== Requirements on the Model
- The model must predict at two quantities per configuration:
  - The phase delay along each principal axis.
  - The phase difference between them.
  - If we want to consider frequency bandwidth then also slope would be  nice.
- Phase must not be regressed directly.
  - It wraps at $360 degree$, so identical geometries would map to both $0 degree$ and $360 degree$ and any regressor would average them toward $180 degree$ at every wrap boundary.
  - Adding more data does not fix this; it makes the model more confidently wrong.
- The model therefore predicts the four components of the two complex principal reflection coefficients:
  $ bold(R)_"major" = a + j b, quad bold(R)_"minor" = c + j d $
  from which every quantity of interest is derived:
  $ phi_"major" = angle(R_"major"), quad Delta phi = angle(R_"minor" dot R_"major"^* ) $
  This is continuous through resonance, bounded by $|R| <= 1$ for a passive structure, and makes $Delta phi$ a derived quantity rather than a second independent target.
- Usually in surrogate model two basic approaches are used - Polynomial Response Surfaces and Gaussian Processes (one of Radial Basis Funcion models, it's more generalized for is Kriging) @Forrester2008
  - Both natively supply confidence intervals, which can be of great help in the optimization process.
  - Both are able to achieve results on relatively small dataset.

== Why Polynomial Response Surfaces Were Discarded
- A polynomial response surface is a global low-order fit $y = bold(X) beta$, attractive because it has no hyperparameters and is fitted by a single linear solve.
- It is inappropriate here because the response is dominated by a resonance.
  - Near resonance the phase slope reaches several hundred degrees per millimetre of geometry, so a low-order polynomial cannot represent the curve at all.
  - Raising the order to capture the resonance makes the number of basis terms grow combinatorially in five dimensions.
- The data requirement then becomes larger than the full-factorial sweep the surrogate was meant to avoid, which defeats the purpose.

== Why Gaussian Processes Were Discarded
- A Gaussian process was the natural alternative: it gives a posterior variance for free, which is the correct confidence interval, and it is designed for small sample sizes.
- Two reasons ruled it out.
  - *Multiple independent models.*
    - A GP regresses one scalar output at a time, so four complex components require four independent GPs.
    - Each fits its own length scales, the four can disagree with each other, and each can independently violate $|R| <= 1$.
    - There is no shared representation, even though the major and minor axes are the same physics seen from two directions.
  - *Sample efficiency is not the binding constraint.*
    - The main advantage of a GP is that it works with a few dozen samples.
    - Here the CST sweep can be run repeatedly overnight and produces on the order of $10^4$ configurations, so that advantage is not needed.
    - At that sample count a neural network matches or exceeds a GP while training faster per sample.
- The Gaussian process is therefore not used. If MLP will not work I will try it or Kriging model or some other modification.

== Multi-Layer Perceptron
- The chosen model is a single multi-layer perceptron with one shared latent representation and five outputs, which resolves the multi-model objection directly -- one training run, one set of hyperparameters, and a network that learns the coupling between the two axes.

=== Input Features
- The raw parameters $(a, "ratio", P_"ratio", epsilon_r, H_"pillar", H_"base")$ are a poor coordinate system for a regressor: they do not scale with frequency and they do not expose the physics.
  - As there is no closed-form solution for the response there isn't an exact set of features which would describe the problem best.
  - They are instead chosen so that the network is handed the quantities that the approximate theory says govern the behaviour, leaving it to learn only the correction.
- All features are normalized to the design frequency, so that the $30$ to $15$~GHz scaling leaves them invariant.

==== Geometric and material features
- *Cell footprint* $P \/ lambda_0$, the periodicity normalised to the design wavelength.
  - Rather self explanatory, size of the UC needs to be captured by some parameter and $P_"ratio"$ is all things consider suited for it.
  - While parameter like Fill Fraction uses $P_"ratio"$ it's not recoverable from it - two cells can share a fill fraction while having different electrical size, and the electrical size sets the mutual coupling between adjacent elements, the grating-lobe threshold, and the lateral cutoff of the modes trapped in the cell. All three shift the phase response.
- *Normalised ellipse size* $a \/ P$, the semi-major axis relative to the cell.
  - The ellipse is described relative to the cell rather than in millimeters because the lateral boundary condition and the coupling to neighbours are governed by how much of the cell the pillar occupies, not by its absolute physical size.
    - For the same reason it also isn't scaled relative to $lambda_g$
  - It is the swept tuning variable, and it is retained directly rather than only through derived quantities because the phase response is not monotonic in it
    - It rises through resonance and then falls, so the network needs the coordinate itself in order to locate the resonance.
    - Of course resonance location is then also heavily affected by the aspect ratio or effective permittivities.
  - At worst this introduces a redundant representation alongside the fill fraction, which is the safer failure mode.
- *Fill fraction* $f = pi a b \/ P^2$, the ellipse area over the cell area.
  - This is the dielectric loading of the cell and the quantity that sets its effective permittivity, so it is much closer to physics than the semi-axis length alone.
  - Note that $P$ here is in millimetres, obtained as $P_"ratio" lambda_0$, so the ratio is dimensionless and invariant under the $30$ to $15$~GHz scaling.
- *Electrical pillar height* $(H_"pillar" \/ lambda_0) sqrt(epsilon_r)$, and the same for the base layer.
  - The reflection phase of a grounded dielectric slab is governed by its electrical height, not its physical height, so normalising by $lambda_0$ and by the refractive index $sqrt(epsilon_r)$ makes the feature scale-free and collapses the two independent variables into the one that matters.
  - The base layer is included separately because it is a structural element whose thickness is not required to scale with the cell.
  - If it is held constant while the frequency halves, the stack is no longer a scaled copy of itself, and the feature has to expose that.
- *Relative permittivity* $epsilon_r$ itself, which sets the $Q$ of the resonator and therefore how sharp the resonance is.

==== Anisotropy features
- *Aspect ratio* $a \/ b$ itself.
  - This is the direct geometric measure of elongation and controls the separation between the major- and minor-axis resonances, and hence the achievable phase difference.
  - It is kept on a linear scale. The physical chain from geometry to phase difference is
    $ a\/b -> "shape" -> N_a, N_b -> epsilon_"eff" -> Delta phi $
    and every step is algebraic or rational, none exponential or logarithmic. There is therefore no mechanism that would make the response logarithmic/exponential in elongation, and a transform would only degrade relative precision in the region where the sampling is densest, namely $a\/b -> 1$.
  - The depolarisation factors $N_a$ and $N_b$ are not carried as separate features.
    - They are deterministic functions of the ratio alone, and their effect on the response enters through the effective permittivities below, so including them as well would add no information.

==== Effective-medium features
- *Motivation.*
  - An anisotropic mixture has a different effective permittivity along each principal axis, and the anisotropy arises purely from the shape of the inclusions rather than from any anisotropy in the constituent materials.
  - The pillar is an isotropic dielectric, so the birefringence of the cell comes entirely from its elliptical cross-section.
  - The purpose of this section is to expose that mechanism to the network as a small number of derived quantities.
- *Step 1: the mixing rule.*
  - In the theory of mixtures, an inclusion placed in a host is depolarised by an amount that depends only on its shape.
  - The depolarisation factor $N_j$ along principal axis $j$ quantifies the fraction of the internal field removed by this effect; if it is zero the internal field equals the external one @Sihvola1999.
  - The factors of any inclusion satisfy $N_a + N_b + N_z = 1$, and for reference a sphere has $1\/3$ along each axis, a disc has $(1, 0, 0)$ and a needle $(0, 1\/2, 1\/2)$ @Sihvola1999.
  - For aligned ellipsoidal inclusions the anisotropic Maxwell-Garnett rule then gives the effective permittivity along axis $j$ as @Sihvola1999
    $ epsilon_("eff",j) = epsilon_m (epsilon_m + (epsilon_p - epsilon_m)(N_j + f(1 - N_j))) / (epsilon_m + (epsilon_p - epsilon_m) N_j (1 - f)) $
    where $epsilon_m$ is the host, $epsilon_p$ the inclusion and $f$ the volume fraction. The rule returns $epsilon_m$ as $f -> 0$, and recovers the familiar spherical form when $N_j = 1\/3$.
  - In this work the host is air ($epsilon_m = 1$), the inclusion is the printed dielectric ($epsilon_p = epsilon_r$), and the volume fraction is the fill fraction $f = pi a b \/ P^2$.
- *Step 2: the depolarisation factors of an elliptical cylinder.*
  - Sihvola gives the factors for ellipsoids, including closed forms for prolate and oblate spheroids, but not for cylinders @Sihvola1999.
  - The spheroid is the wrong shape class here anyway: a spheroid is a closed surface of revolution, whereas the pillar has a constant elliptical cross-section and flat ends.
  - The correct shape class is the elliptical cylinder, for which @Zhang2015 give the transverse factors
    $ N_a = b\/(a + b), quad N_b = a\/(a + b), quad N_z = 0 $
    with $N_a + N_b + N_z = 1$. The factors are named after the semi-axis they belong to rather than after a coordinate direction, because the ellipse is rotated by $45 degree$ relative to the array axes and the usual $x, y, z$ labels would be ambiguous here.
  - Zhang and Wu write the rule in an implicit, shape-weighted form rather than the explicit form used above:
    $ (epsilon_("eff",j) - epsilon_m) / (N_j epsilon_("eff",j) + (1 - N_j) epsilon_m) = f (epsilon_p - epsilon_m) / (N_j epsilon_p + (1 - N_j) epsilon_m) $
    Rearranging for $epsilon_("eff",j)$ yields the explicit expression given in step 1, so the two are the same relation. They state explicitly that in the quasi-static limit their result is exactly the Maxwell-Garnett rule @Zhang2015.
  - Two checks confirm the identification.
    - The circular limit $a = b$ returns $(1\/2, 1\/2, 0)$, consistent with Sihvola's statement that a circle corresponds to a depolarisation factor of $1\/2$ @Sihvola1999.
    - The thin-strip limit $b -> 0$ returns $(0, 1, 0)$, consistent with the needle limit @Sihvola1999.
  - Physically the behaviour is the expected one: for a pillar elongated along $a$, a field applied along $a$ is barely depolarised, so $N_a$ is small, while a field across the minor axis sees $N_b -> 1$.
  - The anisotropy of the cell enters through $N_a != N_b$ alone. In the source's notation the factors associated with the surrounding cavity cancel for a square lattice, so no anisotropy is contributed by the lattice itself @Zhang2015.
- *Step 3: the resulting features.*
  - $epsilon_("eff,major")$ and $epsilon_("eff,minor")$, computed with $N_a$ and $N_b$ respectively.
  - Both depend on the fill fraction and therefore on the cell size: for a fixed $a$ and $b$, a larger cell means a smaller $f$, which pulls both values toward unity.
- *Where the model is an approximation.* The construction is an analogy used to generate features, not a predictive model of the cell, and the network is expected to learn the corrections. Three departures from the assumptions are worth naming.
  - *Dilute, separated inclusions.* Maxwell-Garnett assumes dilute inclusions embedded in a three-dimensional host @Sihvola1999, whereas the fill fraction here reaches a few tenths and the pillar is a single connected object rather than a dispersion of separate particles.
  - *The ground plane.* The structure is backed by a conductor, so it is a grounded slab rather than an inclusion in an unbounded host. This changes the absolute reflection phase but not the lateral anisotropy.
  - *The dielectric substrate.* The base layer of thickness $H_"base"$ sits beneath the pillar and is not part of the mixture model at all. This is the more significant omission, but it affects both polarisations in the same way, so it does not alter the anisotropy that these features are meant to capture. Rather than complicate the mixing rule with a second layer, the base thickness is carried as its own input feature and the network learns its effect.
  - Accordingly the effective permittivities should be read as physically motivated coordinates that encode the *shape-driven anisotropy* of the cell, not as quantitative predictions of its permittivity tensor.

==== Summary
#align(center)[
  #table(
    columns: 4,
    align: (left, left, left, left),
    [*Feature*], [*Expression*], [*Units*], [*Purpose*],
    [`p_ratio`], [$P \/ lambda_0$], [--], [electrical size of the cell; coupling, grating lobes],
    [`a_over_P`], [$a \/ P$], [--], [normalised ellipse size; locates the resonance],
    [`fill_fraction`], [$pi a b \/ P^2$], [--], [dielectric loading of the cell],
    [`h_pillar_electrical`], [$(H_"pillar"\/lambda_0)sqrt(epsilon_r)$], [--], [electrical height of the pillar],
    [`h_base_electrical`], [$(H_"base"\/lambda_0)sqrt(epsilon_r)$], [--], [electrical height of the base],
    [`ratio`], [$a\/b$], [--], [geometric elongation; drives $Delta phi$],
    [`eps_r`], [$epsilon_r$], [--], [resonator Q],
    [`eps_eff_major`], [MG with $N_a$, $f$], [--], [loading seen by the major axis],
    [`eps_eff_minor`], [MG with $N_b$, $f$], [--], [loading seen by the minor axis],
  )
]

=== Outputs
- The network predicts the real and imaginary parts of both principal reflection coefficients at the design frequency, plus one frequency-domain descriptor:
  $ bold(y) = [ "Re"(R_"major"), "Im"(R_"major"), "Re"(R_"minor"), "Im"(R_"minor"), dot(phi)_"major" ] $
  - $dot(phi)_"major"$ is the major-axis phase slope in $"deg"\/"GHz"$, obtained as the gradient of a least-squares linear fit of the unwrapped phase across the simulated band.
    - A linear fit is used rather than a two-point difference because the band is narrow and finite differencing would amplify numerical noise.
  - The slope is the standard proxy for element bandwidth: what limits a reflectarray is the *variation* of phase with frequency, not its absolute value.
  - Band spread and a resonance-in-band flag are deliberately *not* targets. (#INFO these are extracted features in the code #TODO describe these later in how the code works)
    - Spread is a consequence of the slope plus curvature, so predicting both would be redundant and potentially inconsistent.
    - Resonance flag is a classification quantity better used to filter samples than to be predicted.
- *Why not a resonance frequency.*
  - Most configurations have no resonance inside the simulated band, so a resonance frequency would be undefined for the majority of samples and defined for a minority.
  - That would require masked losses and would bias training toward the resonant subset, which is exactly the kind of ragged target the complex-coefficient parameterisation was chosen to avoid.

=== Architecture and Training
- *Architecture.*
  - Roughly three hidden layers of width $128$, with a smooth activation (SiLU, ELU or GELU) and a linear output layer.
  - Depth beyond four rarely helps for tabular regression with of order ten inputs, and a smooth activation is preferred over ReLU because a discontinuous derivative can produce visible kinks in the predicted phase curves.
  - No dropout.
    - Don't really need redundant representation, and while it could be used for uncertainty estimations, that information comes from a deep ensemble.
    - Deep ensemble requires each member to be a well-trained deterministic function; dropout noise would confound the ensemble spread, bias each member toward underfitting, and make the calibration check meaningless.
  - No batch normalisation.
    - Its running statistics are estimated on the training distribution and misbehave on inputs far from it, which is precisely the extrapolation regime that matters here.
    - Inputs and outputs are z-scored instead, with the scaling fitted on the training folds only.
    - As the Network will be rather small learning issues which BN solves shouldn't arise.
  - #TODO here there maybe should be some citations, even though it's all rules of thumb I've learned over time idk.
- *Loss.*
  - The reflection coefficients at the design frequency are the primary objective and are weighted more heavily than the auxiliary slope term:
  $ cal(L) = "MSE"_4 (bold(y), hat(bold(y))) + w_"slope" "MSE"(dot(phi), hat(dot(phi))) + lambda_"pass" "mean"("ReLU"(|hat(R)| - 1)^2) $
  - Because all targets are z-scored before the loss is evaluated, $w_"slope"$ is directly interpretable as the relative weight of the slope term, and a small value such as $0.1$ keeps it as a soft regulariser rather than letting a quantity measured in $"deg"\/"GHz"$ dominate.
  - The passivity term is zero whenever $|hat(R)| <= 1$ and therefore cannot distort well-behaved predictions;
     - It only acts where the model would otherwise predict gain, a failure mode an unconstrained regression can exhibit near resonance.
     - Maybe redundant, i don't think model will predict gain outside of couple first epochs. And maybe even there not.
  - $L_2$ regularisation is applied implicitly as decoupled weight decay in the AdamW optimiser rather than as a separate loss term, which is equivalent in effect but interacts better with the adaptive step size. Thus loss function above is only data loss.
- *Early stopping* on the validation loss, with each ensemble member allowed to stop at a different epoch.
  - The model is small, so checkpoints are cheap to retain.
- *Hyperparameters* are tuned with Optuna @Akiba2019, which uses a define-by-run search space and pruning of unpromising trials.
  - The search covers depth, width, activation, learning rate, weight decay, batch size and $w_"slope"$, the latter on a logarithmic scale because it spans two decades.
  - Dropout is excluded by design.
  - During hyperparameters tuning no less then 5-fold cross validation will be used to asses performance of the hyperparameters, since the dataset is not that large.
- *Sampling*
  - For additional training points uses a Sobol sequence rather than a full grid, to correctly sample in the resonance region.
  - TODO: This isn't really applicable for current task. I would need to automate sweep process to run sweeps with correct sampling. Now the dataset is just garbage, even though histograms don't look that bad.
- *Why not a sequence model.*
  - Predicting the whole phase curve over $a$ at once, for instance with a recurrent network, was considered and rejected.
  - There is no causality along $a$, so the ordering assumption is not physically motivated; an autoregressive error would compound at the resonance where the phase slope is steepest; and the design search requires point queries, which a sequence model can only answer by generating a full curve.
  - The curve structure is already exploited implicitly, since neighbouring values of $a$ give neighbouring feature vectors and therefore similar predictions.
  - In case frequency behavior would need to be analyzed better it would be interesting to use neural operators (Fourier NO, DeepONet).
- Every candidate the surrogate proposes is verified by a full CST simulation before it is accepted.
  - The surrogate therefore only needs to rank candidates correctly, not to be accurate in absolute terms.
  - #TODO? Automate the verification loop later, maybe since the model will hopefully be close enough we can just do a+ratio grid search near the solution and converge in idk 10 simulation runs.
    - Will be needed in the future, if there isn't mismatch between model and reality more then couple degree.
    - Maybe then there will be some trend that could be correct idk?

=== Uncertainty Quantification
- Two different uncertainties are needed, and they are not interchangeable.
  - *Generalisation error* answers how accurate the model is overall.
    - It is estimated by $k$-fold cross-validation: the data are split into $k = 5$ folds, the model is trained $k$ times with one fold held out each time, and the spread of the held-out error across folds gives the variance of the performance estimate.
      - Final model is trained on the whole dataset, k-fold gives Generalisation error intervals.
  - *Per-prediction uncertainty*
    - answers how far a single prediction can be trusted.
    - Cross-validation cannot supply this, because it says nothing about any individual input.
- For design work the second is the one that matters:
  - the surrogate has to indicate which regions of the parameter space it has not learned, so that unverified proposals are not trusted and new simulations are directed where they are most informative.
- The method used is a *deep ensemble*: $M = 5$ networks trained from independent random initialisations with independently shuffled data ordering, so that the ensemble members differ only in their starting point and the order in which they see the same data.
  $ mu(bold(x)) = 1/M sum_(m=1)^M f_m (bold(x)), quad sigma^2 (bold(x)) = 1/M sum_(m=1)^M (f_m (bold(x)) - mu (bold(x)))^2 $
  This gives an epistemic uncertainty: it is small where the training data are dense and large where the model is extrapolating, which is exactly the behaviour required for active learning. It is preferred over Monte-Carlo dropout, which is cheaper but tends to produce overconfident intervals, and over a variational Bayesian network, whose tuning cost is not justified at this scale. #TODO pick some random paper on deep ensemble and cite it.
- The uncertainty is propagated through the nonlinearity rather than approximated.
  - Since the design quantity is $Delta phi$, which is a function of all four predicted components, the ensemble spread is taken over $Delta phi$ computed from each member:
  $ sigma_(Delta phi) (bold(x)) = "std"_m [ angle(R_"minor"^((m)) dot R_"major"^((m))^* ) ] $
  rather than by linear propagation of the per-component variances.
- The intervals are validated by a *calibration* check:
  - predictions are binned by predicted $sigma$, and for each bin the fraction of true values falling within $plus.minus 1.96 sigma$ is compared with the nominal $95 thin %$.
  - Systematic under-coverage would indicate overconfident intervals and require recalibration before the uncertainty is used to guide sampling.

=== Physics-Informed Neural Networks
- A physics-informed neural network (PINN) embeds the residual of a governing differential equation in the training loss @Raissi2019:
  $ cal(L) = cal(L)_"data" + lambda cal(L)_"PDE" $
  where $cal(L)_"PDE"$ penalises violation of the PDE at collocation points, evaluated by differentiating the network output with respect to its spatial or temporal coordinates.
- *This formulation does not apply to the present problem.*
  - The network here maps design parameters to a response, $ (a, "ratio", epsilon_r, H) |-> bold(R) $.
  - There are no spatial coordinates in the input, so there is no differential operator to apply and $cal(L)_"PDE"$ is undefined.
  - Writing Maxwell's equations in terms of $d bold(R) \/ d a$ is not a physics residual.
- The distinction is definitional rather than a matter of degree: a model trained purely by supervised learning is not a PINN, regardless of how much physics enters its inputs. The loose reading of the original framework has blurred this.
- Two further properties of PINNs work against their use here.
  - They are formulated for a single instance of a PDE, that is one set of boundary conditions and material parameters, whereas the requirement is a map over a five-dimensional parameter space.
  - And their training is expensive and sensitive to architecture, sampling and optimization settings, which would be paid for no benefit.
- *What is adopted instead* is the approach used in electromagnetic surrogate modelling, where physical consistency comes from the prediction basis rather than from a PDE term.
  - #TODO @Jensen2026
  - For lossless periodic structures the constraint is imposed by projecting the output onto a manifold that enforces energy conservation @Prehn2026.
  - The analogue here is the pair of measures already described: predicting complex reflection coefficients instead of wrapped phase, and penalizing $|R| > 1$ in the loss.
- A neural operator, which does map between function spaces and generalizes across parameters where a PINN does not, would become relevant only if the model were extended from a single frequency to a full frequency sweep. It is not used at present.

== Dataset
=== Scope of the sweep
- The sweep spans five parameters: the cell size `P_ratio` (as a fraction of $lambda_0$), the relative permittivity `epsilon_r`, the pillar height `H_pillar`, the ellipse aspect ratio `ratio = a/b`, and the semi-major axis `a` that serves as the tuning variable.
- Most of the data were computed for `P_ratio = 0.4` at $epsilon_r = 4.5$ and $epsilon_r = 7.5$, the two nominal values bracketing the Zetamix material range.
  - These are nominal figures only; the printed permittivity has to be measured before any of it is trusted, and further simulations will follow that measurement, hopefully with proper sampling.
- A smaller set extends the coverage to $epsilon_r = 7$ and to the larger cell size $P_ratio = 0.5$, again at $epsilon_r = 4.5$ and $epsilon_r = 7.5$.
- In total 19 596 configurations were computed.

=== Structure of the data
- The 19 596 configurations collapse into 342 distinct geometries, each swept along `a` at a fixed (P_ratio, epsilon_r, H_pillar, ratio).
  - Per-geometry sample counts: minimum 29, median 52, maximum 94.
- `a` spans 0.4 mm to 3.192 mm over the whole dataset.
- Distinct values swept per parameter:

#align(center)[
  #table(
    columns: 4,
    align: (left, center, center, left),
    stroke: none,
    inset: (x: 6pt, y: 4pt),
    [*Parameter*], [*Values*], [*Count*], [*Range or set*],
    [`P_ratio`], [2], [2], [0.4, 0.5],
    [`epsilon_r`], [3], [3], [4.5, 7, 7.5],
    [`H_pillar`], [28], [28], [2.2 mm to 12 mm],
    [`ratio`], [15], [15], [1.0 to 2.4],
  )
]

  #align(center)[
  #figure(
    image("../simulations/data/statistics/counts_summary.pdf", width: 46%),
    caption: [Valid samples per (P_ratio, epsilon_r) combination, after applying the 0.01 mm validity clearance. The two `P_ratio = 0.4` permittivities carry the bulk of the data; `P_ratio = 0.5` and `epsilon_r = 7` are represented by much smaller sets and act as coverage checks rather than training material.],
  )
]

- The height axis is the most densely sampled of the four fixed parameters, with 28 values between 2.2 mm and 12 mm.
  - Sampling is regular in 0.4 mm steps up to 5.4 mm, then finer between 5.4 mm and 7.5 mm where the resonance moves most rapidly with height, then coarser again.
  - Height range for each configuration was chosen based on the results of the preliminary study - only promising regions for given $epsilon_r$ were simulated.
- `ratio` is sampled at every tenth between 1.0 and 2.4, which gives a monotone trade between anisotropy and the achievable phase offset.
- `P_ratio` and `epsilon_r` carry the least coverage by design: they are fabrication choices rather than continuous design variables, so they are explored at a handful of discrete operating points instead of being swept.

=== Geometric validity
- Not every simulated configuration is physically realisable.
  - The ellipse major axis is rotated $45 degree$ relative to the array axes, so its axis-aligned bounding box has half-extent $sqrt((a^2 + b^2)\/2)$ with $b = a slash "ratio"$.
  - Requiring that box to stay inside the cell gives the largest usable semi-major axis
    $ a_max = P/2 dot sqrt(2)\/sqrt(1 + 1\/"ratio"^2) $
  - A larger `ratio` therefore permits a longer major axis, since the same elongation that lengthens the ellipse also shrinks its bounding box.
    - A sweep that shares one upper limit across several ratios consequently never reaches the geometric edge for the elongated configurations, or for some configurations computes unrealizable unit cells.
- Applying a validity clearance of 0.01 mm removes 876 configurations, leaving *18 720 valid samples* (95.5 %).
  - This clearance is purely a safeguard against CST approximation of elipse with some polygon, where I cannot be sure that the approximation didn't cross UC boundary, even with valid `a` and `b`.
  - It is deliberately much smaller than the 0.2 mm manufacturability clearance used when assessing printability, because a cell whose ellipse nearly touches the wall is still a valid design point and its response is worth having in the training set.
- Every one of the 342 geometries reaches the geometric edge:


#align(center)[
  #figure(
    table(
      columns: 6,
      align: center,
      stroke: none,
      inset: (x: 6pt, y: 4pt),
      table.header(
        [*`P_ratio`*], [*`epsilon_r`*], [*valid `n`*], [*`a_min`* (mm)], [*`a_max`* (mm)], [*distinct `ratio`*],
      ),
      [0.4], [4.5], [9 250], [0.400], [2.584], [15],
      [0.4], [7.0], [952],   [0.400], [2.209], [4],
      [0.4], [7.5], [5 406], [0.400], [2.578], [15],
      [0.5], [4.5], [2 802], [0.400], [3.138], [11],
      [0.5], [7.5], [310],   [0.400], [2.478], [1],
    ),
    caption: [Valid sample counts and `a` coverage per (P_ratio, epsilon_r) combination. `a_max` differs between rows because it is set by the elongation: the longest ellipse that still fits the cell grows with `ratio`, and each combination contains a different mix of ratios.],
  )
]

=== Coverage of the tuning variable
- For a surrogate to learn the phase behaviour it must see `a` distributed across the interval where the response changes, not clustered at one end.
  - The histograms below pool all ratios within a (P_ratio, epsilon_r) combination, since the question is how densely the tuning axis is covered for a given cell and permittivity, independently of which elongation produced each point.
- The two heavily sampled combinations, $epsilon_r = 4.5$ and $epsilon_r = 7.5$ at `P_ratio = 0.4`, cover the interval continuously from 0.4 mm to above 2.5 mm.
  - The histogram is not flat: the density falls towards large `a`, because the higher-ratio configurations reach the geometric limit earlier and contribute fewer points to the upper end.
  - This is a property of the geometry rather than of the sampling plan, and it is the reason the `a_max` column differs between rows.
- The $epsilon_r = 7$ set covers only 2.209 mm and just four ratios, so its upper `a` range is truncated relative to the other 0.4 cells. It is adequate to check whether the permittivity trend interpolates between 4.5 and 7.5, but not to train across the full interval on its own.
- The `P_ratio = 0.5`, $epsilon_r = 7.5$ row is a single ratio at a single cell size. It is a spot check on the larger cell, not a coverage set, and its `a` histogram is correspondingly a single cluster rather than a distribution.

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(image("../simulations/data/statistics/a_hist_P0.4_er4.5.pdf", width: 100%), caption: [`a` coverage, `P_ratio` 0.4, $epsilon_r$ 4.5.]),
  figure(image("../simulations/data/statistics/a_hist_P0.4_er7.5.pdf", width: 100%), caption: [`a` coverage, `P_ratio` 0.4, $epsilon_r$ 7.5.]),
)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(image("../simulations/data/statistics/a_hist_P0.4_er7.pdf", width: 100%), caption: [`a` coverage, `P_ratio` 0.4, $epsilon_r$ 7.0.]),
  figure(image("../simulations/data/statistics/a_hist_P0.5_er4.5.pdf", width: 100%), caption: [`a` coverage, `P_ratio` 0.5, $epsilon_r$ 4.5.]),
)


=== Coverage of the aspect ratio
- `ratio` takes few discrete values, so the histograms place one bar per value rather than binning continuously. That makes the count at each elongation directly readable, which is what shows whether a combination was swept across the full anisotropy range or only at one end.

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(image("../simulations/data/statistics/ratio_hist_P0.4_er4.5.pdf", width: 100%), caption: [`ratio` coverage, `P_ratio` 0.4, $epsilon_r$ 4.5.]),
  figure(image("../simulations/data/statistics/ratio_hist_P0.4_er7.5.pdf", width: 100%), caption: [`ratio` coverage, `P_ratio` 0.4, $epsilon_r$ 7.5.]),
)

- The dominant sets span the full 1.0 to 2.4 range in 15 steps.
  - Coverage is not uniform across the range either: the low-ratio end carries more samples per value, since those configurations have the larger usable `a` interval and therefore more samples in it.
- The sparse combinations are not uniform in `ratio` at all: $epsilon_r = 7$ covers four values and `P_ratio = 0.5` at $epsilon_r = 7.5$ covers one.
  - For the surrogate this matters more than the raw sample count, because a combination sampled at a single ratio cannot teach the model how the response depends on elongation, only what it is at that one elongation.

=== Limitations of the dataset
- *Low `a` carries little design information.*
  - Below roughly 1.2 mm the phase difference between the two principal axes stays small, so those configurations cannot satisfy the quadrature condition and are of limited use beyond establishing the low-delay baseline.
  - Still that regions is sampled quite well.
- *The permittivity axis is represented at three nominal values only.*
  - The printed permittivity will differ from the nominal figure and will vary with infill.
  - So a measured value between 4.5 and 7.5 would fall in a gap where the surrogate is interpolating with no direct evidence.
- *`P_ratio` is sampled at two points.*
  - That is enough to detect a strong trend with cell size but not to fit a smooth dependence on it.
  - It's possible `P_ratio=0.5` will be removed anyway, if they won't contribute to the accuracy, since UC size of $0.4 lambda_0 $ is more premising anyway.
- *The sampling is not space-filling.* The `a` values within a geometry are regularly spaced, and the four fixed parameters are swept on a coarse grid rather than on a Sobol sequence.
  - For the current purpose this is acceptable, since the grid is denser in the resonance region than a generic space-filling design would be.
  - It does mean the dataset is not as good as it should be, and the uncertainty estimates should be read with that in mind.
  - Automating the sweep so that it can be driven by the surrogate's own uncertainty is the natural next step, and is what the Sobol sampling discussed above is intended for.


#pagebreak()

#v(2em)
#bibliography("bibliography.bib", style: "ieee")

