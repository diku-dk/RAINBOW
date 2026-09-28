# Literature review: interactive deformable-body simulation

This review surveys influential computer-graphics work on deformable and soft
body simulation from the 1980s onward, with emphasis on ACM SIGGRAPH, ACM
SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), Eurographics,
ACM Transactions on Graphics, and IEEE TVCG. It also includes selected surveys,
tutorial-style material, and open-source systems that clarify how the methods
are used in practice.

The review is organized around five questions:

1. What time-stepping or state-update method is used?
2. How is the deformable object represented?
3. What material model and parameters are used?
4. How are collisions detected?
5. How are contact and penetration resolved?

“Interactive” is used carefully. In this literature it can mean a user can
manipulate the model while it runs, a simulation reaches roughly frame-rate
feedback, or a haptic loop has a much tighter deadline. These are different
regimes. A paper may demonstrate interactive manipulation for a reduced model
while its high-resolution offline simulation is not interactive.

This is a curated design review, not a bibliometric claim about the complete
field. The evidence ledger favors primary papers and implementation material
that expose enough of the algorithm to answer the questions above. Counts in
the design tables below are counts in this curated sample, not estimates of
the fraction of all published work. “Not reported” is retained deliberately:
it prevents a frequently repeated graphics convention from being mistaken for
an experimentally established material or solver choice.

## 1. Scope and current SOFT capabilities

The current implementation is a contact-free, first-order tetrahedral FEM
prototype. Its state is nodal position and velocity; its elements are linear
tetrahedra with lumped mass; its constitutive choices are Saint
Venant--Kirchhoff and stable Neo-Hookean elasticity; and its stepping choices
are semi-implicit Euler, implicit backward Euler, implicit midpoint,
trapezoidal, and average-acceleration Newmark. The implicit position solve is
matrix-free L-BFGS with Armijo-style globalization and fallback, watchdog, and
rescue mechanisms. NumPy and JAX backends share the force/residual model, with
backend-specific execution paths.

The prototype does not yet implement collision detection, contact, friction,
damping as a material mechanism, adaptive timestep control, cutting, remeshing,
higher-order elements, reduced coordinates, or a sparse direct/Newton solver.
The design question is therefore not “does the literature use FEM?” but “which
parts of the FEM and nonlinear-solve design space are deliberately covered,
and which omissions matter for the intended next extension?”

The implementation map is maintained in the [SOFT architecture](architecture.md),
[numerics](numerics.md), [time-stepping](time-stepping.md), and
[extensions](extensions.md) pages. This review supplies the external evidence
used to assess those choices.

## 2. Executive synthesis

- The field contains two durable solver families. Mechanics-oriented work
  commonly uses implicit Euler or another implicit scheme and spends the
  budget on a linear or nonlinear solve. Graphics-oriented real-time work
  commonly uses position/constraint updates with a bounded iteration count,
  accepting controlled inexactness for predictable frame time.
- “Soft body” is not one discretization. The literature uses spring particles,
  triangle cloth meshes, tetrahedral finite elements, subdivision volumes,
  reduced modal coordinates, deformation textures, and constraint graphs.
- Contact is usually a separate subsystem. Broad-phase and narrow-phase
  collision detection produce candidate pairs; response is then handled by
  penalty/repulsion forces, impulses, geometric projection, Lagrange
  multipliers, complementarity, augmented Lagrangians, or combinations of
  these.
- Implicit integration permits larger stable steps, but it does not by itself
  make a simulation interactive. The cost of assembling, solving, and
  repeatedly updating contact constraints becomes the limiting factor.
- Interactive systems often optimize bounded latency and visual plausibility
  rather than tightly converged trajectories. Fixed iteration budgets,
  warm-starting, reduced coordinates, local updates, and GPU parallelism are
  recurring ways of making that compromise explicit.
- Material parameters are often scenario parameters rather than measured
  physical constants. Papers emphasize stiffness ratios, damping, constraints,
  and visual behavior; exact Young's modulus, density, Poisson ratio, and
  damping values are frequently omitted or tuned per example.

The review then moves from reference settings to the literature evidence
ledger and the assessment of SOFT's design choices. Focused literature
case studies come next, followed by thematic method families, collision/contact
methods, material practice, and interaction regimes. The final part collects
framework, library, and author case studies before stating the implications
for SOFT.

## 3. Reference parameter settings

The literature is much better at naming a method than at publishing a
reproducible parameter block. The table below therefore separates three kinds
of evidence: exact values visible in the source or implementation, values
reported in the paper but only for a representative example, and settings
that are explicitly *not reported*. “Not reported” is useful information: it
means that a value should not be reverse-engineered from a plot or presented as
a field-wide default.

The SOFT rows are included as an operational reference for reproducing the
framework benchmark. They are not claims about the literature.

### Aggregate tetrahedral parameter overview

This is the compact reference table. It aggregates the quantitative evidence
from the tetrahedral subset rather than repeating paper-specific algorithms.
Ranges are ranges reported by the cited work or group of works; they are not
recommended universal defaults. `NR` means that the source does not report a
reliable numerical value in the material reviewed. Normalized quantities and
SI quantities are kept distinct.

| Family or representative use | Materials | Young's modulus $E$ | Poisson ratio $\nu$ | Density $\rho$ | Damping | Time step | Iteration or solver limit | Tetrahedral size |
|---|---|---:|---:|---:|---|---|---|---:|
| **SOFT scientific baseline** | StVK or stable Neo-Hookean | $100\,\mathrm{kPa}$ | $0.49$ | $1100\,\mathrm{kg/m^3}$ | None in the material model | $10^{-4}\,\mathrm{s}$ reference; candidates through $10^{-1}\,\mathrm{s}$ | L-BFGS max iterations $5$--$150$; history $4$--$12$; absolute tolerance $10^{-1}$--$10^{-8}$; relative tolerance $5\times10^{-1}$--$10^{-6}$ | $594$ linear tets in the canonical grid |
| **Interactive corotational / linear-tet FEM** | Corotational linear elasticity; StVK in some examples | $50$--$100\,\mathrm{kPa}$ in the GI fracture examples; other works are NR | $0.33$ | $1000\,\mathrm{kg/m^3}$ in the GI examples; one real-time work uses normalized $\rho=1$ | Lumped damping matrices or tuned damping; coefficient NR | $10\,\mathrm{ms}$ in *Stable Real-Time Deformations*; other examples NR | Conjugate-gradient or reduced/modal solves; iteration caps NR | $440$--$1000$ tets in reported examples |
| **Small-step interactive benchmark** | Linear constitutive tetrahedral FEM | $10\,\mathrm{MPa}$ | $0.45$ | $1000\,\mathrm{kg/m^3}$ | No separate material damping reported; damping is method-dependent | One frame step, or one frame divided into $100$ substeps; fixed SI $\Delta t$ NR | $1$ substep/$100$ iterations, $100$ substeps/$1$ iteration, or backward-Euler PCG with $1$ substep; per-step cap is method-dependent | $12{,}800$ tets |
| **Medical / local tetrahedral FEM** | StVK and local elastic-position formulations | $4$--$10\,\mathrm{kPa}$ in the reported liver example | NR | NR | NR | NR | Local iterative solver; iteration limit NR | $596$ tets |
| **Stable flesh / near-incompressible hyperelasticity** | Stable Neo-Hookean, corotational comparison | NR in accessible source; $\nu=0.488$ reported for a representative example | $0.488$; discussion includes $\nu\ge 0.4$ | NR | NR | NR | Newton/CG with positive-semidefinite Hessian treatment; exact cap NR | Tetrahedral dynamics are discussed; representative published count is $45{,}809$ elements, but the main flesh example is hexahedral |
| **Robust quasistatic and implicit tetrahedral FEM** | Invertible FEM, robust flesh, hyperelastic and constitutive variants | NR | NR | NR | NR | Quasistatic or implicit; fixed $\Delta t$ often not the reported control parameter | Newton--Raphson, modified/filtered stiffness, CG, or contact-constrained solves; caps/tolerances NR | Scene-dependent; NR in the accessible reports |
| **Reduced tetrahedral / haptic FEM** | StVK and reduced nonlinear elasticity | NR | NR | NR | NR | Approximately $1\,\mathrm{ms}$ haptic loop in representative reduced-order work | Reduced dimension and haptic loop budget dominate; full-order iteration cap NR | Original mesh counts are scene-dependent; reduced dimension is the reported limit |

The aggregate table should be read as a map of reported practice, not as a
claim that the field uses a common material calibration. In particular,
iteration limits and damping coefficients are often omitted even when $E$ and
$\nu$ are given. The next table preserves the per-work evidence and records
those omissions explicitly.

### Source-specific records for traceability

| Source or use case | Representation and material settings | Density, damping, and loading | Time-step settings | Nonlinear/linear solve settings | Contact and interactive regime |
|---|---|---|---|---|---|
| **SOFT canonical baseline** ([baseline definition](baselines.md)) | Linear tetrahedra; beam $0.10\,\mathrm{m}$ long with $0.02\,\mathrm{m}$ square section; $E=100\,\mathrm{kPa}$, $\nu=0.49$; StVK or stable Neo-Hookean. The equivalent isotropic constants are $\mu=33.56\,\mathrm{kPa}$ and $\lambda=1.644\,\mathrm{MPa}$ | $\rho=1100\,\mathrm{kg/m^3}$; no material damping; bending gravity $(0,-9.81,0)\,\mathrm{m/s^2}$; amplified loads are selected by the example | Scientific reference: $\Delta t=10^{-4}\,\mathrm{s}$, normally $1000$ steps or a user-selected duration; candidate sweep includes up to $0.1\,\mathrm{s}$ | Scientific sweep defaults: max iterations $5,10,20,30,50,75,100,150$; history $4,8,12$; absolute tolerances $10^{-1}$ through $10^{-8}$; relative tolerances $5\times10^{-1}$ through $10^{-6}$; line search and backtracking/watchdog globalization are swept | No collision or contact; scientific acceptance uses trajectory/energy error, while interactive acceptance uses finite and bounded motion plus latency |
| **SOFT scientific implicit BFGS reference run** | Same as above; matrix-free residual and directional-residual strategy are backend-independent at the model level | Same; no damping | $\Delta t=10^{-4}\,\mathrm{s}$ | Representative accepted profile: max iterations $30$, history $12$, absolute tolerance $10^{-6}$ or the looser profile selected by the sweep, relative tolerance $10^{-6}$, Armijo backtracking, max line-search iterations $24$; the latest permissive run accepted $10^{-3}$ absolute tolerance with $3.81$ mean iterations and $1.98\%$ trajectory error | No contact; this is a convergence/accuracy study, not an interactive graphics test |
| **SOFT interactive autotune** | Same canonical linear-tet soft body and constitutive choices | Same; no material damping is added by the acceptance test | Same $10^{-4}\,\mathrm{s}$ fine reference; candidate steps are swept through $0.1\,\mathrm{s}$ | Same configurable BFGS/globalization sweep; inexact convergence is allowed if motion remains finite, bounded, and responsive | Explosion guards: energy envelope, displacement, velocity, and per-frame displacement; trajectory agreement is diagnostic rather than the sole acceptance criterion |
| Baraff--Witkin, *Large Steps in Cloth Simulation* ([SIGGRAPH 1998](https://doi.org/10.1145/280814.280821)) | Triangular cloth; stretch, shear, bending, and damping energies; exact cloth constants are scene-specific | Damping is part of the cloth model, but the accessible paper summary does not provide one universal parameter block | Implicit Euler with adaptive step-size logic; exact default step sequence is not a field-wide constant | Modified constrained conjugate-gradient solve; exact iteration/tolerance values are implementation- and scene-dependent | Collision constraints are central; interactive CPU cloth, with bounded progress more important than a single exact trajectory |
| Barbič--James, *Real-Time Subspace Integration* ([project page](https://graphics.cs.cmu.edu/projects/stvk/)) | StVK deformable model projected to a low-dimensional subspace; reduced cubic force expressions | Exact material constants are example-dependent and not treated as universal defaults | Implicit Newmark; reported dynamics rates reach the kilohertz haptic regime because online cost depends on reduced dimension | Precomputation plus reduced online evaluation; the reduced dimension is the key practical setting, not a large full-order iteration budget | Reduced collision processing and haptic contact; hard real-time interaction |
| Smith--de Goes--Kim, *Stable Neo-Hookean Flesh Simulation* ([Dynamic Deformables](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/)) | Structured hexahedral/lattice representation; stable Neo-Hookean compared with corotational elasticity; representative example uses $\nu=0.488$, $45{,}809$ elements and $156{,}078$ DoFs | Other exact material constants are example-specific; consult the published comparison before reusing them | Exact production timestep and convergence tolerances are not a universal setting reported by the accessible summary | Positive-semidefinite Hessian treatment and robust nonlinear solve; exact iteration caps vary by scene | Large-deformation flesh/graphics regime; contact and scene integration are part of the broader Dynamic Deformables material |
| Teran--Sifakis--Irving--Fedkiw, *Robust Quasistatic FEM* ([SCA 2005](https://graphics.cs.wisc.edu/Papers/2005/TSIF05/)) | Tetrahedral FEM for character flesh; robust elastic force evaluation under inversion and compression | Exact Young's modulus, density, damping, and load values are scene-specific/not reported in the review-accessible summary | Quasistatic solve rather than a single dynamic-$\Delta t$ recipe | Newton--Raphson; modified positive-definite stiffness enables conjugate gradients; projection/regularization policy is more important than a single tolerance | Collision and self-collision in a production flesh pipeline; robustness-oriented rather than a standalone frame-rate benchmark |
| Otaduy et al., *Implicit Contact Handling for Deformable Objects* ([paper](https://media.disneyanimation.com/uploads/production/publication_asset/32/asset/EG2009_implicit_contact.pdf)) | Deformable mechanics plus inequality contact constraints; exact constitutive settings vary by scene | Exact material, damping, and friction values are not a universal reported block | Implicit constrained dynamics; exact $\Delta t$ is scene- and application-dependent in the accessible paper description | Iterative constraint anticipation and a mixed complementarity/LCP solve; exact tolerances and iteration caps are implementation settings | Rolling, sliding, stacking, and impact; contact-rich production/interactive regime |
| Rivers--James, *FastLSM* ([SIGGRAPH/TOG 2007](https://doi.org/10.1145/1276377.1276480)) | Embedded geometry in regular lattices with shape matching; effective stiffness depends on lattice/shape-matching parameters rather than a directly calibrated FEM $E,\nu$ pair | Damping and material-like parameters are tuned for robust deformation; no universal SI parameter block | Real-time update; exact step size is application-dependent and not a reusable default | Linear-time fast summation and robust shape matching; bounded-cost interaction is the objective | Dozens of soft bodies in interactive/game-like settings; strict continuum fidelity is intentionally traded for speed |
| **SOFA / HOBAK / VegaFEM / SuperDex / Drake** | These systems expose multiple materials and representations, but parameter values are scene files or examples rather than universal defaults | Density, damping, and friction are configuration data; do not infer them from the framework name | Integrator and $\Delta t$ are configured per scene; no single system-wide value | Solver type, tolerances, iteration caps, and preconditioners are configuration-dependent; see the system sections below | These are implementation references for contact-rich or robotics regimes, not directly comparable benchmark rows |

For practical reuse, the most defensible starting point is the canonical
SOFT block, followed by a sensitivity sweep over $E$, $\nu$, density,
load, and $\Delta t$. The literature does not justify copying a single
“standard soft-body” parameter set into a different mesh, scale, or contact
regime.

## 4. Literature evidence ledger

The table below is a design-decision ledger, not a chronology. The rows are
grouped by the question they help answer—representation, integration,
robustness, contact, reduction, or architecture—and the publication year is
retained only to locate the work historically. Read the settings table above
when an exact parameter reference is needed; read the ledger below for the
algorithmic role and the evidence each work contributes.

The table summarizes representative works. “Not reported” means that the
paper or accessible implementation does not provide enough information for a
reliable classification; it should not be interpreted as “not used.”

### Tetrahedral work-level parameter evidence

This ledger is the detailed companion to the aggregate table in Section 3.
Each row gives the range or limit that can be supported from the cited work.
`NR` is deliberate: it prevents a value from being mistaken for a field-wide
default when the paper leaves it to the scene, source code, or supplementary
material. A range is used when a work reports multiple scenes or parameter
sets.

| Work | Tetrahedral representation and size | Material evidence | $E$ range or value | $\nu$ range or value | $\rho$ range or value | Damping evidence | Time-step evidence | Iteration / solver limit | Evidence qualification |
|---|---|---|---:|---:|---:|---|---|---|---|
| SOFT canonical baseline ([baseline definition](baselines.md)) | Linear tet grid; $594$ tets | StVK or stable Neo-Hookean | $100\,\mathrm{kPa}$ | $0.49$ | $1100\,\mathrm{kg/m^3}$ | None | $10^{-4}\,\mathrm{s}$ reference; sweep to $10^{-1}\,\mathrm{s}$ | L-BFGS $5$--$150$ iterations; history $4$--$12$; absolute tolerance $10^{-1}$--$10^{-8}$; relative tolerance $5\times10^{-1}$--$10^{-6}$ | Directly reproducible SOFT setting, not literature evidence |
| Müller et al., *Stable Real-Time Deformations* ([paper](https://www.cs.rpi.edu/~cutler/publications/stable_real_time_deformations.pdf)) | Volumetric tube; $1000$ tets | Corotational / volumetric linear FEM | Source table reports values, but the accessible extraction does not preserve the modulus units reliably | $0.33$ | Normalized $\rho=1$ in the reported example | Lumped inertia and damping matrices; coefficient NR | $10\,\mathrm{ms}$ | Conjugate-gradient solve; iteration cap NR | The source reports normalized units; do not convert $\rho=1$ to SI |
| Müller et al., *Real-time simulation of deformation and fracture of stiff materials* ([GI 2004 paper](https://matthias-research.github.io/pages/publications/GI2004.pdf)) | Snake $440$, Cow $970$, Dragon $834$ tets | Linear elastic/plastic material | $50$--$100\,\mathrm{kPa}$ | $0.33$ | $1000\,\mathrm{kg/m^3}$ | Plastic creep $0$--$100\,\mathrm{s^{-1}}$; plastic-yield settings are also reported | NR | Solver iteration cap NR | Values come from the paper's material table; yield stress is $60\,\mathrm{kPa}$ for the Dragon and infinite for the other two examples |
| Macklin et al., *Small Steps in Physics Simulation* ([SCA 2019 paper](https://diglib.eg.org/server/api/core/bitstreams/5ccd0aa4-edd9-48a1-8ed5-1c50d227a37a/content)) | Cantilever; $12{,}800$ tetrahedral FEM elements | Linear constitutive model | $10\,\mathrm{MPa}$ | $0.45$ | $1000\,\mathrm{kg/m^3}$ | No separate material damping reported | One frame step, or $100$ substeps per frame; fixed SI frame $\Delta t$ NR | XPBD: $1$ substep/$100$ iterations and $100$ substeps/$1$ iteration; backward Euler PCG: $1$ substep | Exact reported comparison: $4$--$12\,\mathrm{ms}$ per frame depending on method and setting |
| Smith, de Goes, and Kim, *Stable Neo-Hookean Flesh Simulation* ([paper](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf)) | Tetrahedral dynamics are included; representative published example has $45{,}809$ elements and $156{,}078$ DoFs, while the main flesh example is hexahedral | Stable Neo-Hookean and corotational comparison | NR | $0.488$; paper discusses near-incompressible $\nu\ge 0.4$ | NR | NR | NR | Newton/CG with positive-semidefinite Hessian treatment; cap NR | Do not transfer the representative element count to the tetrahedral subset without checking the specific example |
| Irving, Teran, and Fedkiw, *Invertible Finite Elements* ([SCA 2004 page](https://graphics.cs.wisc.edu/Papers/2004/ITF04/)) | Linear tetrahedral FEM; scene-specific count | Invertible elastic formulation | NR | NR | NR | NR | Implicit; fixed value NR | Robust force evaluation; Newton/CG details and caps NR | Numeric material settings are not reported in the accessible abstract/page |
| Teran, Sifakis, Irving, and Fedkiw, *Robust Quasistatic FEM* ([SCA 2005 page](https://graphics.cs.wisc.edu/Papers/2005/TSIF05/)) | Tetrahedral flesh FEM; scene-specific count | Robust elastic force evaluation | NR | NR | NR | NR | Quasistatic; no fixed dynamic $\Delta t$ | Newton--Raphson with modified positive-definite stiffness and CG; caps NR | The work is evidence for solver safeguards, not for a reusable parameter block |
| Sifakis et al., *Arbitrary Cutting of Deformable Tetrahedralized Objects* | Background tetrahedral mesh; count NR | Elastic FEM with cutting | NR | NR | NR | NR | Progressive cutting; dynamic step NR | Robust update during cuts; iteration cap NR | Geometry/topology settings are emphasized over material calibration |
| Otaduy et al., *Implicit Contact Handling for Deformable Objects* ([paper](https://media.disneyanimation.com/uploads/production/publication_asset/32/asset/EG2009_implicit_contact.pdf)) | Deformable solids/cloth; tet count NR | Scene-dependent deformable mechanics | NR | NR | NR | NR | Implicit constrained dynamics; $\Delta t$ NR | Iterative constraint anticipation and MLCP; tolerances/caps NR | Contact settings are the reported contribution; material block is not universal |
| Barbič and James, *Real-Time Subspace Integration* ([project page](https://graphics.cs.cmu.edu/projects/stvk/)) | Full mesh plus reduced subspace; tet count NR in the accessible summary | StVK | NR | NR | NR | NR | Approximately $1\,\mathrm{ms}$ haptic loop | Reduced dimension and precomputation are the limits; nonlinear full-order cap NR | This is evidence for reduced-order budgeting, not a full-order tet parameter reference |
| Bargteil and Cohen, *Quadratic Bézier Finite Elements* | Quadratic and linear tetrahedra; count NR | Corotational linear strain | NR | NR | NR | NR | Dynamic FEM; $\Delta t$ NR | Adaptive degree and precomputation; iteration cap NR | Relevant to element-order choice; numeric material values are not in the accessible summary |
| Liver tetrahedral local-position example ([source](https://pmc.ncbi.nlm.nih.gov/articles/PMC3966138/)) | $596$ tetrahedral elements | StVK/local elastic formulation | $4$--$10\,\mathrm{kPa}$ | NR | NR | NR | NR | Local iterative solve; cap NR | Medical/surgical example, included as a tetrahedral parameter reference rather than a graphics default |

| Work | Venue/year | Representation and material | Time stepping / solve | Collision detection | Contact resolution | Interactive regime |
|---|---|---|---|---|---|---|
| Terzopoulos, Platt, Barr, Fleischer, *Elastically Deformable Models* | SIGGRAPH 1987 | Deformable curves, surfaces, and solids; elasticity-based models | Numerical solution of coupled dynamic equations; implementation details reflect early continuum/mechanical simulation | Impenetrable obstacles and applied constraints | Force/constraint treatment in the deformable-model formulation | Primarily animation/offline; foundational rather than a modern real-time system |
| Terzopoulos and Fleischer, *Deformable Models* | The Visual Computer 1988 | Curves, surfaces, and solids; elastic and inelastic extensions | Dynamic differential equations; implicit methods are part of the early deformable-model lineage | Obstacles and external constraints | Forces and constraints; inelastic reference-state evolution for viscoelasticity/plasticity/fracture | Animation and modeling; interactive manipulation is a goal, but no modern frame-budget claim |
| O'Brien and Hodgins, *Graphical Modeling and Animation of Brittle Fracture* | SIGGRAPH 1999 | Volumetric finite elements with fracture-dependent topology | Explicit dynamic update in the fracture animation pipeline | Geometric fracture and newly exposed surfaces | Element separation and topology changes rather than persistent contact mechanics | Offline animation; important because topology change is a separate problem from ordinary deformation |
| Müller et al., *Stable Real-Time Deformations* lineage | SIGGRAPH/ACM TOG 2002--2004 | Volumetric tetrahedral FEM, commonly linear tetrahedra with corotational or modal acceleration | Explicit or reduced/modal stepping in interactive examples; solver details vary by paper | Collision proxies and application-specific geometric tests | Penalty, constraints, or application-specific response; no single unified contact method | Interactive reduced-resolution regime; the key gain is reducing the dynamic state, not making full nonlinear FEM cheap |
| Irving, Teran, and Fedkiw, *Invertible Finite Elements for Robust Simulation of Large Deformation* | SCA 2004 | Linear tetrahedral FEM with an invertibility-preserving elastic formulation | Implicit dynamics and robust force evaluation around degenerate or inverted configurations | Geometry-dependent; element robustness is the central contribution | Stabilized elastic forces and collision handling in the surrounding simulator | Robustness-oriented mechanics; a building block for larger-step solvers, not itself a contact architecture |
| Teran, Sifakis, Irving, and Fedkiw, *Robust Quasistatic Finite Elements and Flesh Simulation* | SCA 2005 | Tetrahedral FEM for skeleton-driven flesh; robust elastic forces under inversion and heavy compression | Quasistatic Newton--Raphson with modified positive-definite stiffness for conjugate gradients | Collision and self-collision treatment is part of the robust flesh pipeline | Collision response coupled to the quasistatic solve and robust force evaluation | Production-oriented character/flesh simulation; a key reference for separating robustness from ordinary timestep accuracy |
| Sifakis, Shinar, Irving, and Fedkiw, *Hybrid Simulation of Deformable Solids* | SCA 2007 | Embedded sample points in conforming or background tetrahedral meshes; separate representations for elasticity, collision, and constraints | Mesh-based FEM for elastic forces with embedded/meshless handling for difficult operations | Embedded representation simplifies collision, plasticity, and fracture without repeated global remeshing | Constraints and contact can use a representation different from the elastic mesh | Strong architectural reference for multi-representation deformable simulation and future SOFT collision/contact design |
| Sifakis, Der, and Fedkiw, *Arbitrary Cutting of Deformable Tetrahedralized Objects* | SCA 2007 | Embedded high-resolution tetrahedral geometry with cuts introduced through a background discretization | Robust FEM update during progressive cutting; avoids restrictive remeshing assumptions | Tool/tissue intersection and arbitrary cuts | Topology change and cut-surface handling rather than ordinary contact response | Interactive surgical/graphics regime; directly relevant to future cutting of linear-tet models |
| Sifakis and Barbic, *FEM Simulation of 3D Deformable Solids* | SIGGRAPH Course 2012; Synthesis Lectures 2015 | Practical reference covering tetrahedral FEM, corotational and hyperelastic materials, invertibility, and model reduction | Discusses implicit integration, CG, multigrid, and reduced-order methods | Collision/contact is treated as part of the broader deformable-solid pipeline | Compares application-dependent contact and constraint strategies | Field-level practitioner reference; useful for checking that SOFT's theory, verification, and autotuning coverage is not too narrow |
| Rojas, Sifakis, and Kavan, *Differentiable Implicit Soft-Body Physics* | ICML 2021 | Energy-based FEM soft bodies with Neo-Hookean elasticity and actuation | Implicit state transitions defined by minimization; implicit differentiation through the solve using matrix-free reverse-mode derivatives | Contact is not the primary contribution | Differentiable energy/force formulation supports control and learning objectives | Differentiable robotics/control regime; directly relevant to extending JAX-based SOFT beyond forward simulation |
| Baraff and Witkin, *Large Steps in Cloth Simulation* | SIGGRAPH 1998 | Triangular cloth mesh; continuum-inspired stretch, shear, bending, and damping | Implicit Euler; modified constrained conjugate gradient solve; adaptive step-size logic | Coherency-based bounding boxes and cloth collision tests | Direct constraint enforcement/position alteration, with constraints maintained during the iterative solve | Interactive-oriented CPU cloth; large steps are the central contribution, but scale and hardware are historical |
| Bridson, Fedkiw, Anderson, *Robust Treatment of Collisions, Contact and Friction for Cloth Animation* | SIGGRAPH/ACM TOG 2002 | Particle/triangle cloth; internal dynamics are deliberately decoupled from collision handling | Compatible with the underlying cloth integrator; collision method is solver-independent | Geometric collision tests with thickness and continuous/robust handling for remaining collisions | Non-stiff repulsion forces plus fail-safe collision processing; static and kinetic friction | Production-animation quality and robust cloth; collision cost, not only integration, limits interactive scale |
| Selle et al., *Robust High-Resolution Cloth Using Parallelism, History-Based Collisions, and Accurate Friction* | IEEE TVCG 2009 | Triangular cloth mesh; edge stretch springs, bending springs, and damping | Modified time integration with distributed parallel evolution; the paper discusses semi-implicit and fully implicit choices rather than reducing the result to one universal integrator label | AABB hierarchy plus history-based proximity tests and collision state | History-based repulsion/attraction, Gauss--Seidel collision response, accurate friction, and rigid impact zones for unresolved collisions | High-resolution/hero-cloth production regime; scales to very large meshes, but is not an ordinary frame-rate interactive method |
| Capell et al., *A Multiresolution Framework for Dynamic Deformations* | SCA 2002 | Volumetric subdivision hierarchy and embedded domains for elastic solids | Implicit solver for large stable timesteps | Not the primary contribution; geometric embedding and constraints support interaction | Constraints on the reduced/hierarchical deformation representation | Interactive-oriented reduced deformation; high-resolution detail is introduced selectively |
| James and Pai, *ArtDefo: Accurate Real Time Deformable Objects* | SIGGRAPH 1999 | Boundary-element model of static linear elasticity; surface representation with physically meaningful material parameters | Fast updates exploit coherent changes in boundary conditions; not a dynamic FEM timestepper | Boundary/contact conditions are central to the interaction model | Boundary integral response and haptic force computation | Early real-time interaction regime; important because it achieves speed through formulation and coherence rather than coarse spring tuning |
| James and Fatahalian, *Precomputing Interactive Dynamic Deformable Scenes* | SIGGRAPH 2003 | Data-driven reduced state-space models of nonlinear deformable scenes, including self-contact examples | Offline impulse-response simulation and low-rank dynamics; runtime playback is extremely cheap | Self-collisions are resolved during precomputation and are implicit at runtime | Contact is baked into the precomputed scene response | Highly constrained interactive regime; important counterexample to treating runtime interaction as online full-order simulation |
| James and Pai, *BD-Tree: Output-Sensitive Collision Detection for Reduced Deformable Models* | SIGGRAPH 2004 | Reduced deformable models with arbitrary displacement bases | Independent of the mechanical timestepper | Bounded Deformation Tree for output-sensitive collision detection | Detection building block; response remains separate | Interactive collision regime; directly relevant to reduced collision geometry for a future SOFT extension |
| Barbič and James, *Real-Time Subspace Integration for St. Venant--Kirchhoff Deformable Models* | SIGGRAPH 2005 | Full-order nonlinear geometry with StVK material projected to a low-dimensional subspace | Precomputed cubic reduced internal forces and stiffness; implicit Newmark subspace integration | BD-Tree and reduced collision processing | Haptic contact forces using the reduced state and sampled geometry | Hard real-time haptics; large meshes can be integrated at kilohertz rates because runtime cost depends mainly on subspace dimension |
| Kry, James, and Pai, *EigenSkin: Real-Time Large Deformation Character Skinning in Hardware* | SCA 2002 | Large nonlinear FEM character deformations compressed into pose-dependent eigenbases | Quasistatic/data-driven reduction; runtime evaluation is hardware skinning rather than online full-order integration | Not a collision solver; deformation is driven by articulated pose | No online contact response | Real-time rendering regime; demonstrates that interactive deformation can move simulation cost offline and leave only low-dimensional evaluation online |
| Rivers and James, *FastLSM: Fast Lattice Shape Matching for Robust Real-Time Deformation* | SIGGRAPH/ACM TOG 2007 | Embedded geometry in regular lattices with shape matching rather than calibrated continuum FEM | Robust position/shape-matching dynamics with linear-time fast summation and damping | Embedded geometry and particle/lattice interaction; application-dependent contact | Shape-matching response, not complementarity contact | Game/interactive regime; deliberately trades strict physical consistency for speed and robustness |
| Nesme, Kry, Jeřábková, and Faure, *Preserving Topology and Elasticity for Embedded Deformable Models* | SIGGRAPH/ACM TOG 2009 | Coarse embedded linear-elastic model preserving disconnected topology, heterogeneous materials, and empty space | Coarse simulation with embedded high-resolution geometry | Embedded representation improves boundary and topological behavior | Application-dependent constraints/contact | Interactive high-resolution deformation; relevant to separating simulation resolution from visual/collision resolution |
| Teschner et al., *Collision Detection for Deformable Objects* | Eurographics STAR 2004 | Survey covering meshes, particles, and deformable surfaces | Not a single integrator | Bounding-volume hierarchies, distance fields, spatial partitioning, and related broad/narrow phases | Survey of penalty, constraints, and application-specific response | Focuses on interactive collision workloads in surgery and entertainment |
| Faure et al., *SOFA, a Multi-Model Framework for Interactive Physical Simulation* | Springer 2012 | Multiple mechanical, collision, visual, and other representations connected through mappings | Components expose differential equations, integration schemes, linear solvers, and simulation-loop policies | Collision models and collision pipelines are separate components | Constraint, penalty, friction, and contact-response components can be composed independently | Interactive medical simulation, robotics, haptics, and research prototyping; architecture prioritizes reuse and algorithm comparison |
| Teschner et al., *A Versatile and Robust Model for Geometrically Complex Deformable Solids* | CGI 2004 | Point-sampled/deformable solids with geometric complexity | Interactive deformable-body update; the contribution is the robust model and geometric handling rather than a single integrator | Spatial structures and geometric queries for complex deformable solids | Contact and collision handling integrated with the geometric model | Interactive surgery/entertainment-oriented regime; important predecessor to later collision architectures |
| Keiser et al., *Contact Handling for Deformable Point-Based Objects* | VMV 2004 | Point-based deformable objects | Point-based dynamics with contact correction | Deformable point-object proximity/contact queries | Contact handling for point-based objects, including response to penetration | Interactive contact regime; relevant as a non-FEM alternative and as a contact design reference |
| Heidelberger et al., *Consistent Penetration Depth Estimation for Deformable Collision Response* | VMV 2004 | Deformable surface/point representations | Independent of the mechanical timestepper | Penetration-depth computation for deformable intersections | Consistent penetration-depth-based response | Interactive collision-response regime; relevant to a future SOFT narrow phase and recovery policy |
| Bielser, Glardon, and Teschner, *A State Machine for Real-Time Cutting of Tetrahedral Meshes* | Pacific Graphics 2003; Graphical Models 2004 | Tetrahedral volumetric meshes with changing topology | Real-time update around mesh surgery | Tool/tissue intersection and cut-surface tracking | Topology changes and element subdivision/separation | Real-time surgical simulation; an early direct reference for future cutting of linear-tet meshes |
| Teschner et al., *Optimized Spatial Hashing for Collision Detection of Deformable Objects* | VMV 2003 | Deformable volumetric/surface geometry | Independent of the mechanical timestepper | Spatial hashing for broad-phase collision candidates | Detection only; response is separate | Interactive collision-detection regime; useful CPU broad-phase baseline |
| Müller et al., *Position Based Dynamics* | VRIPHYS/Eurographics 2006 | Particles and geometric constraints; material behavior encoded through constraints | Position update with iterative constraint projection; velocity layer can be omitted | Collision constraints generated from geometric proximity/intersection | Sequential or parallel projection to valid positions; penetration is removed directly | Strongly interactive: bounded iterations provide predictable cost and controllability |
| Müller et al., *Meshless Deformations Based on Shape Matching* | SIGGRAPH/ACM TOG 2005 | Particle clusters with local shape matching; no connectivity-dependent FEM mesh is required | Explicit goal-position/shape-matching update | Particle and cluster proximity tests | Position correction and cluster-based response | Strong interactive regime; stable and simple, but material calibration is less direct than continuum FEM |
| Müller, *Hierarchical Position Based Dynamics* | VRIPHYS 2008 | Hierarchies of particles/constraints for deformable objects | Hierarchical iterative position projection | Hierarchical collision constraints | Position projection across multiple levels | Interactive regime with a route to accelerating convergence and large-scale manipulation |
| Müller and Chentanez, *Solid Simulation with Oriented Particles* | SIGGRAPH/ACM TOG 2011 | Oriented particles carrying local rotational and deformation information | Particle-based explicit/iterative dynamics | Particle-based proximity tests | Particle constraints and collision response | Interactive solid simulation; relevant alternative to tetrahedral FEM for graphics-oriented robustness |
| Müller et al., *Strain Based Dynamics* | SCA 2014 | Particle-based deformation controlled directly by strain constraints | Iterative strain-based update rather than a conventional global FEM solve | Particle collision constraints | Position/strain projection | Interactive regime; direct strain control is useful for graphics authoring but is not equivalent to calibrated continuum FEM |
| Müller et al., *Air Meshes for Robust Collision Handling* | SIGGRAPH/ACM TOG 2015 | Deformable surface meshes augmented with an auxiliary collision mesh | Independent of the underlying integrator | Auxiliary “air mesh” improves detection robustness and temporal coherence | Robust collision response using the auxiliary representation | Interactive/production collision regime; relevant to separating collision geometry from simulation geometry |
| Macklin et al., *Non-Smooth Newton Methods for Deformable Multi-Body Dynamics* | SIGGRAPH/ACM TOG 2019 | Hyperelastic tetrahedral FEM coupled to rigid bodies and frictional contact | Non-smooth Newton iteration for nonlinear complementarity problems; symmetric linear systems are the inner building block | Contact geometry for coupled deformable/rigid systems | Frictional contact and coupling solved in a unified nonsmooth formulation | Interactive robotics regime; directly relevant to a future contact-capable implicit SOFT solver |
| Müller et al., *Physically Based Shape Matching* | Computer Graphics Forum 2022 | Meshless particle groups derived from continuous constitutive models | Shape-matching update with a physically motivated constitutive interpretation | Particle proximity tests | Shape matching and particle contact response | Interactive regime; bridges heuristic shape matching and physically interpretable material parameters |
| Macklin et al., *Small Steps in Physics Simulation* | SCA 2019 | Particle and rigid/deformable constraint systems | Trades large solver iterations for many small substeps; relevant to the interactive timestep/iteration budget | Collision constraints evaluated across substeps | Constraint projection and collision response repeated at small substeps | Interactive real-time regime; important comparison for SOFT's choice between larger implicit steps and substepping |
| Macklin and Müller, *A Constraint-Based Formulation of Stable Neo-Hookean Materials* | MIG 2021 | Constraint-based stable Neo-Hookean material for particle/position-based simulation | Constraint projection rather than global FEM Newton; designed for robust large deformation | Collision constraints in the same position-based framework | Compliant/constraint response | Interactive graphics regime; a material/solver alternative, not a replacement for calibrated FEM |
| Hu et al., *A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling* | SIGGRAPH/ACM TOG 2018 | MLS-MPM particles and background grid with CPIC displacement discontinuities | Particle/grid update; avoids mesh entanglement during large deformation and fracture-like separation | Grid/particle and rigid-body coupling | Displacement discontinuity and two-way rigid coupling | High-performance graphics regime; strong alternative when topology change and extreme deformation dominate |
| Hu et al., *ChainQueen: A Real-Time Differentiable Physical Simulator for Soft Robotics* | ICRA 2019 | Differentiable MLS-MPM for soft robots | Forward and reverse simulation through analytical/autodiff-compatible operations | Contact-capable particle/grid simulation | Contact and actuation integrated with soft-robot control/design optimization | Real-time soft-robotics and differentiable-control regime; accuracy is evaluated together with gradient quality |
| Hu et al., *Taichi: A Language for High-Performance Computation on Spatially Sparse Data Structures* | SIGGRAPH Asia 2019 | Data-oriented sparse data structures and GPU/CPU kernel programming; not a material model | Runtime/compiler substrate for MPM, FEM, and differentiable simulation implementations | Provides data structures and kernels; collision policy remains application-specific | Application-specific | Execution-platform contribution; relevant to implementation productivity and GPU scalability, not a standalone timestepper |
| Hu et al., *DiffTaichi: Differentiable Programming for Physical Simulation* | ICLR 2020 | Differentiable physical simulation programs, including elastic-object MPM | Source-transformed reverse-mode differentiation through simulation code | Application-specific contact handling | Application-specific | Differentiable simulation regime for control, inverse design, and system identification |
| Galoppo et al., *Fast Simulation of Deformable Models in Contact Using Dynamic Deformation Textures* | SCA 2006 | Layered deformable bodies with a rigid/articulated core and surface deformation textures | Implicit, parallelizable formulation with large steps | Two-stage, output-sensitive proximity queries on dynamic deformation textures | Constraint forces using Lagrange multipliers and approximate implicit integration | Designed for high-resolution contact and real-time shading; SIMD/commodity-hardware oriented |
| Barbič and James, *Time-Critical Distributed Contact for 6-DoF Haptic Rendering of Adaptively Sampled Reduced Deformable Models* | SCA 2007 | Reduced deformable models with adaptively sampled surface points | Fast reduced-state time stepping followed by contact evaluation | Sphere-tree / pointshell hierarchy and adaptive sampling | Time-critical contact force estimation for haptics | Haptic regime: contact must meet a much tighter loop than ordinary graphics frame rate |
| Otaduy et al., *Implicit Contact Handling for Deformable Objects* | Eurographics/CGF 2009 | Deformable solids and cloth; heterogeneous deformables | Implicit constrained dynamics with large timesteps | Collision/contact detection feeding inequality constraints | Implicit complementarity constraints solved with iterative constraint anticipation and an MLCP solver | Interactive-oriented robust contact, including stacking, rolling, sliding, and high-speed examples |
| Miguel and Otaduy, *Efficient Simulation of Contact between Rigid and Deformable Objects* | 2011 | Rigid bodies coupled to deformable FEM or cloth models | LCP formulation with partitioned constraints and modified projected Gauss--Seidel | Rigid/deformable contact with large coupled contact regions | Efficient frictional/contact constraint solve exploiting rigid/deformable block structure | Contact-intensive interactive simulation; directly relevant to future rigid--soft coupling in SOFT |
| Garre and Otaduy, *Haptic Rendering of Objects with Rigid and Deformable Parts* | Computers & Graphics 2010 | Deformable tool model with rigid and flexible components | Robust implicit integration for the deformable tool | Self-collision and contact between deformable objects/tools | Haptic force rendering through a coupled rigid handle and deformable state | Haptic/interactive regime; emphasizes force-feedback stability and latency |
| Miguel et al., *Modeling and Estimation of Energy-Based Hyperelastic Objects* | Computer Graphics Forum 2016 | Hyperelastic objects with energy models estimated from deformation data | Configuration-dependent fitting and simulation of nonlinear energy models | Contact is not the main contribution | Energy-based material response suitable for data-driven parameter estimation | Data-driven material-modeling regime; relevant to validation and calibration rather than timestep stability alone |
| Mercier-Aubin, Winter, Kry, and Levin, *Adaptive Rigidification of Elastic Solids* | SIGGRAPH/ACM TOG 2022 | Elastic solids with regions adaptively switched between deformable and rigid representations | Runtime adaptive model simplification based on local motion and deformation | Contact is application-dependent; rigidification changes the effective collision workload | Contact response remains compatible with the adaptive mechanical representation | Interactive performance regime; relevant to spending nonlinear-solve work only where deformation actually changes |
| Cai, Coevoet, Jacobson, and Kry, *Active Learning Neural C-space Signed Distance Fields for Reduced Deformable Self-Collision* | Graphics Interface 2022 | Reduced deformable models with learned configuration-space collision distances | Reduced-order dynamics paired with learned collision queries | Neural C-space SDFs approximate self-collision for reduced models | Learned collision constraints/queries, with accuracy controlled by active sampling | Interactive reduced-order collision regime; relevant to future fast self-collision for SOFT model reduction |
| Shinar, Schroeder, Fedkiw, *Two-way Coupling of Rigid and Deformable Bodies* | SCA 2008 | Rigid bodies plus deformable bodies, including thin shells and arbitrary constitutive models | Unified time-integration and two-way coupled algorithms | Rigid/deformable and self-collision handling | Coupled contact, stacking, friction, and constraints | Interactive-capable coupled simulation; complexity grows with contact and model resolution |
| Bargteil and Cohen, *Animation of Deformable Bodies with Quadratic Bézier Finite Elements* | ACM TOG 2014 | Quadratic Bézier tetrahedral elements, optionally mixed adaptively with linear elements; corotational linear strain | Dynamic FEM with adaptive degree and efficient precomputation; exact timestep details are implementation-dependent | Standard geometric collision mechanisms; not the contribution | Application-dependent contact response | Interactive-oriented higher-order FEM; improves rest-shape/detail efficiency but increases element and integration complexity |
| Bender et al., *Position-Based Simulation of Continuous Materials* | Computers & Graphics 2014 | Continuum-based strain and bending energies on cloth and volumetric bodies; anisotropy and elastoplasticity | Iterative position-based energy reduction rather than force-based Newton dynamics | Geometric proximity and constraint tests | Position projection with material constraints; examples include degenerate/inverted elements | Explicitly interactive: thousands of degrees of freedom with controllable, bounded iterations |
| Bouaziz et al., *Projective Dynamics* | SIGGRAPH/ACM TOG 2014 | FEM-inspired constraints for solids, cloth, shells, and examples | Implicit Euler reformulated as alternating local projections and a prefactorized global solve | Application-dependent collision constraints; core method focuses on elastic constraints | Projection-based constraints in an optimization framework | Strong interactive regime: reports 49k DoFs at about 3.1 ms per iteration and 10 iterations per frame in a representative example |
| Macklin, Müller, Chentanez, *XPBD* | MIG 2016 | Particles and compliant constraints representing elastic and dissipative potentials | Position-based iterative solve with compliance; time-step/iteration dependence is reduced | Collision constraints in the same position-constraint framework | Compliant constraint projection with persistent constraint behavior | Interactive real-time regime; quality and stiffness are traded against iteration budget |
| Smith, de Goes, and Kim, *Stable Neo-Hookean Flesh Simulation* | SIGGRAPH/ACM TOG 2018 | Hexahedral lattice and hyperelastic Neo-Hookean family; near-incompressible flesh with Poisson ratio near 0.5 | Quasistatic/dynamic implicit Newton with CG; element Hessians are projected to positive semidefinite form | Collision is application-dependent; the contribution is constitutive and solver robustness | Solver-compatible constraints/contact in the surrounding character simulator | High-fidelity production/offline character simulation; one example has 45,809 elements and 156,078 DoFs, with 25.6 s per average timestep |
| Kim and Eberle, *Dynamic Deformables: Implementation and Production Practicalities* | SIGGRAPH Courses 2020/2022 | Unified cloth and 3D-solid production systems; HOBAK examples include tetrahedral FEM and hyperelastic materials | Practical implicit integration, Newton--Raphson, Newmark, and illustrative BDF1/BDF2 implementations; production choices are distinguished from teaching code | Proximity queries, continuous collision detection, and global intersection analysis are treated as separate tools | Production collision response for cloth, solids, and two-way coupling; failure handling is a first-class concern | Production animation regime: throughput, robustness, and visual failure resistance across complex shots |
| Lin, Chitalu, and Komura, *Isotropic ARAP Energy Using Cauchy--Green Invariants* | ACM TOG 2022 | Isotropic ARAP/corotational energy written in terms of Cauchy--Green invariants | Closed-form derivatives and Hessian; Newton-type implicit integration with positive-semidefinite filtering | Not the contribution; can be combined with a standard deformable collision pipeline | Solver-compatible elastic forces and constraints | Interactive-oriented constitutive and solver acceleration; reports up to 3.5x speedup over rotation-factorization alternatives |
| Chitalu, Dubach, and Komura, *Binary Ostensibly-Implicit Trees for Fast Collision Detection* | Eurographics/CGF 2020 | Compact canonical binary BVHs; representation is independent of material and mesh discretization | Not a timestepper | Memory- and bandwidth-efficient broad phase for deformable geometry | Detection only; candidate pairs are passed to a separate narrow/contact resolver | Interactive collision infrastructure, especially useful when memory traffic limits broad-phase throughput |
| Chitalu, Dubach, and Komura, *Bulk-Synchronous Parallel Simultaneous BVH Traversal for Collision Detection on GPUs* | I3D 2018 | Multiple dynamic BVHs traversed in parallel | Not a timestepper | GPU broad phase with topology-centered expansion and balanced work | Detection only; response is a separate subsystem | GPU interactive/throughput regime; reports up to 7.1x improvement over a streams-based traversal model |
| Chitalu, Miao, Subr, and Komura, *Displacement-Correlated XFEM for Simulating Brittle Fracture* | Computer Graphics Forum 2020 | XFEM enrichment plus an explicit crack surface and half-edge cutting; quasi-static LEFM | Quasi-static fracture solve | Cut geometry and crack-surface construction rather than ordinary self-contact | Crack-front propagation and local mesh cutting | Interactive/surgical-style topology-change extension; avoids global remeshing but adds enrichment and cut integration |
| Fan, Chitalu, and Komura, *Simulating Brittle Fracture with Material Points* | ACM TOG 2022 | MPM particles, local continuum damage mechanics, Voronoi crack surfaces, and rigid fragments | MPM-style dynamic updates; local damage formulation avoids a global linear solve | Impact/contact triggers damage and supplies forces to the fracture model | Damage evolution, velocity discontinuity, explicit crack surfaces, and rigid-fragment conversion | Destruction and visual-effects regime; a topology-change alternative to baseline FEM, not a replacement for ordinary soft-body contact |
| Li et al., *Deformable Objects Collision Handling with Fast Convergence* | Eurographics/CGF 2015 | Deformable objects with large deformation; elastic energy formulation | Optimization derived from implicit integration | Collision inequalities, with redundant constraints pruned | General linear constraints solved with an MPRGP-derived method | Interactive/stable large-deformation target; reports an order-of-magnitude speedup over ICA in its experiments |
| Li et al., *Incremental Potential Contact* | SIGGRAPH/ACM TOG 2020 | FEM and other discretizations; contact-safe incremental potential formulation | Implicit Euler expressed as an incremental potential with barrier terms and line search | Continuous collision detection through barrier activation and distance queries | Barrier-based non-penetration and friction without penalty stiffness | Robust large-step contact; moderate models may be interactive, but contact and nonlinear iterations remain dominant |
| Schneider et al., *Poly-Spline Finite-Element Method* and PolyFEM | ACM TOG 2019; open-source library | High-order triangles/tetrahedra, splines, polygons/polyhedra; PolyFEM supports multiple constitutive models and orders beyond linear | Library-level transient integrators and nonlinear solves; configuration-dependent rather than one prescribed method | Problem-dependent collision and boundary handling | Problem-dependent constraints/contact; a reference architecture for moving beyond linear-tet-only assumptions | Scientific/engineering and research-prototyping regime; flexibility and accuracy are prioritized over one fixed real-time path |
| Ton-That, Kry, and Andrews, *Generalized eXtended Finite Element Method for Deformable Cutting via Boolean Operations* | Computer Graphics Forum 2024 | XFEM enrichment decouples cut geometry from the background simulation mesh | FEM integration with discontinuous integration over cut subdomains | Cut geometry represented by robust mesh Booleans, not only collision pairs | Discontinuous enriched basis and local integration preserve deformation across cuts | Interactive/surgical-extension regime; local cut cost avoids global remeshing but adds enrichment and integration complexity |
| Fernández-Fernández, Löschner, and Bender, *Progressively Projected Newton's Method* | Computer Graphics Forum 2026 | Element-based hyperelastic deformables, including contact-rich examples | Newton with selective Hessian projection; compared with Projected Newton and Project-on-Demand Newton over timesteps, tolerances, and resolutions | Contact-rich tests use the surrounding deformable-contact pipeline | Projected/conditional Hessian strategies provide descent directions for nonlinear solves | State-of-the-art solver regime; emphasizes fewer eigendecompositions and faster convergence, not a new material or contact model |
| Chen et al., *Vertex Block Descent* | SIGGRAPH/ACM TOG 2024 | Volumetric elastic bodies, including linear tetrahedral FEM | Variational implicit Euler solved by vertex-level Gauss--Seidel/block descent; mesh coloring and GPU parallelism | Parallel collision processing integrated with the solver | Constraint, collision, and friction formulations; bounded iterations retain stability | Modern GPU interactive-to-large-scale regime; some reported scenes are seconds per step, while smaller scenes target real-time use |
| Giles, Diaz, Yuksel, *Augmented Vertex Block Descent* | SIGGRAPH/ACM TOG 2025 | Elastic bodies and coupled rigid/deformable systems | VBD plus augmented-Lagrangian updates; implicit-Euler target | GPU collision processing | Hard constraints, stacking, friction, and attachments through augmented Lagrangians | GPU real-time/stable regime, including very large contact scenes; performance depends strongly on hardware and iteration budget |

## 5. Assessment of SOFT design choices

### Spatial discretization: linear tetrahedra are representative, but not sufficient

Linear tetrahedra are a defensible first implementation for SOFT. They are
the common baseline in graphics FEM because they have constant reference-space
shape gradients, cheap element force evaluation, simple assembly, and a direct
path to lumped mass. They also make matrix-free residual evaluation and NumPy/
JAX parity comparatively straightforward. The drawbacks are equally important:
linear elements represent curved rest shapes poorly, can be stiff or locking-
prone in nearly incompressible regimes, and require many elements to resolve
smooth bending or detailed geometry.

In the curated volumetric-FEM subset of this review, linear tetrahedra are the
dominant *baseline*, not the dominant endpoint. They appear explicitly in the
early interactive FEM and invertibility work, in many corotational and
hyperelastic graphics implementations, and in modern GPU solvers such as VBD.
The counterexamples are decisive: Smith--de Goes--Kim use a hexahedral lattice
for flesh; Bargteil--Cohen use quadratic Bézier tetrahedra; PolyFEM supports
high-order triangles and tetrahedra, splines, and polygonal/polyhedral bases;
reduced and embedded methods avoid integrating every rendered degree of
freedom. Therefore the current choice is suitable for a reference kernel, but
the architecture should not make linear tetrahedra the only future element
family.

| SOFT spatial choice | Evidence in reviewed work | Assessment and consequence |
|---|---|---|
| Linear tetrahedral elements | Common in interactive volumetric FEM, invertibility, corotational FEM, and VBD baselines | Keep as the verification baseline; expose element kernels so higher order, hex, shell, or mixed elements can be added without changing time-stepping interfaces |
| Lumped mass | Common in graphics-oriented explicit and semi-implicit implementations; convenient for matrix-free updates | Appropriate for the current prototype, but compare against consistent mass before making accuracy claims for higher-order elements or modal reduction |
| Higher-order tetrahedra / Bézier elements | Bargteil--Cohen, Weber et al., PolyFEM, and p-multigrid work | Important extension for curved rest shapes and accuracy per element; requires quadrature, richer state, better preconditioning, and a policy for contact geometry |
| Hexahedral or lattice elements | Smith--de Goes--Kim and production character work | Important for near-incompressible flesh and structured character lattices; not a reason to replace linear tets in the reference implementation |
| Reduced, embedded, or texture representations | Capell, Galoppo, Barbič--James, modal/subspace work | Main route to interactive high-resolution deformation; changes what “degrees of freedom” and collision geometry mean |

### Material models: StVK is representative as a baseline, not as a universal flesh model

Saint Venant--Kirchhoff is widely used in graphics because it is simple,
quadratic in Green strain, easy to differentiate, and useful for testing
finite-element assembly and nonlinear time integration. It is not robustly
physical at arbitrarily large strain: the quadratic energy can become
non-convex and can produce nonphysical behavior under extreme deformation.
That makes the current StVK implementation useful for verification and
controlled moderate-strain tests, but not a sufficient constitutive model for
contact-rich or highly compressed soft tissue.

Stable Neo-Hookean is also representative, especially for flesh and large
deformation. Smith, de Goes, and Kim explicitly target the nearly
incompressible regime and report Poisson ratios near 0.5; their example uses
ν = 0.488 and compares the proposed stable model against corotational
elasticity. The important lesson is not that ν = 0.488 is a universal setting,
but that graphics flesh work deliberately operates close to incompressibility
and therefore needs a stable volumetric response and a solver that can handle
the resulting conditioning.

| Material family | Where it is representative | Typical evidence or settings | Relevance to SOFT |
|---|---|---|---|
| StVK / Green-strain elasticity | Baseline nonlinear FEM, moderate strain, educational and reference implementations | Often selected for simple parameterization; exact physical parameters are frequently scene-specific or omitted | Keep for verification, patch tests, and controlled nonlinear studies; label it as a model limitation in large-strain examples |
| Corotational linear elasticity | Real-time volumetric graphics and interactive FEM | Uses linear strain in a rotating local frame; efficient constant element stiffness is a major attraction | Useful comparison model, but current SOFT should not infer that corotational and finite-strain hyperelasticity are interchangeable |
| Stable Neo-Hookean | Flesh, high rotation, large deformation, near-incompressible tissue | Smith et al. use ν = 0.488 in a representative flesh comparison and analyze positive-semidefinite Hessian projections | Strong current choice; add parameterized tests over ν, shear modulus/Young's modulus, and inversion states |
| Inversion-safe anisotropic hyperelasticity | Fiber-reinforced or direction-dependent materials and badly conditioned elements | Kim, de Goes, and Iben derive closed-form eigensystems, an inversion-safe anisotropic invariant, and element rehabilitation | Important if SOFT grows toward tissue, fibers, cutting, or robust extreme deformation; keep constitutive kernels separate from solver policy |
| Mooney--Rivlin, Arruda--Boyce, Fung, anisotropic models | Biological tissue, rubber-like materials, anisotropic tissue, advanced production | Kim et al. compare several hyperelastic families; Bender et al. support anisotropy and elastoplasticity | Future constitutive plug-in interface; validate stress, tangent, and conditioning independently of the stepper |
| XPBD compliance / constraint materials | Interactive graphics where stiffness must be controllable under bounded iterations | Compliance absorbs timestep/iteration dependence rather than representing a unique continuum modulus | Relevant only if SOFT adds a constraint-oriented interactive branch; do not silently compare compliance to Young's modulus |

The literature does not support a single “typical” numerical material value
for generic soft bodies. Density, Young's or shear modulus, Poisson ratio,
damping, thickness, and friction are normally chosen from the application:
cloth, flesh, rubber, surgical tissue, and animation props have different
scales. A useful SOFT benchmark should therefore publish dimensionless
ratios or nondimensional conditioning indicators alongside SI values, for
example stiffness-to-inertia ratio, ν, timestep relative to the fastest mode,
and contact thickness relative to element size.

### Time stepping and nonlinear solvers: the current coverage is good but misses the dominant contact solvers

The current SOFT time-stepping coverage is unusually good for a small
reference implementation: semi-implicit Euler, backward Euler, implicit
midpoint, trapezoidal, and average-acceleration Newmark cover first-order
stable dynamics, second-order low-dissipation dynamics, and a structural-
dynamics formulation. The same position/residual machinery feeds the
matrix-free L-BFGS nonlinear solve.

The field coverage is not complete. The most important omissions are not
another named one-step formula, but alternative nonlinear and contact solve
families:

| Family | Representative works | What it adds beyond current SOFT |
|---|---|---|
| Newton / Projected Newton | Teran et al.; Smith--de Goes--Kim; Longva et al.; Fernández-Fernández et al. | Exact or projected Hessian information, quadratic local convergence, and explicit handling of indefinite element Hessians |
| Project-on-Demand / Kinetic Newton | Longva et al., *Pitfalls of Projection* | Robustness only where needed, avoiding the cost and convergence damage of projecting every element Hessian |
| Progressively Projected Newton | Fernández-Fernández, Löschner, Bender | A current state-of-the-art projected-Newton variant; uses residual information to project a small subset of element Hessians |
| Projective Dynamics | Bouaziz et al. | Local/global split and prefactorized global matrix; attractive when energies have the required projective form |
| Vertex Block Descent / AVBD | Chen et al.; Giles et al. | Local block updates, mesh coloring, GPU execution, and augmented-Lagrangian hard constraints |
| Complementarity / nonsmooth Newton | Otaduy et al.; Macklin et al.; Deul et al. | Contact-aware solves for non-penetration, friction, and changing active sets |
| Barrier-based incremental potentials | Li et al., *Incremental Potential Contact* | Continuous collision-safe contact without tuning a penalty stiffness; couples CCD, barrier energy, friction, and line search |
| Higher-order FEM and p-multigrid | Bargteil--Cohen; PolyFEM; cubic-element/p-multigrid work | More accuracy per element and a route to curved geometry, at the cost of richer quadrature and preconditioning |

Thus, SOFT has broad *time-integrator* coverage but narrow *nonlinear
solver/contact* coverage. Adding BDF2 or generalized-α would be valuable for
specific numerical-damping and multistep studies, but it is less urgent for
field coverage than adding a sparse Newton/projected-Newton comparison and a
contact-safe incremental-potential path.

### Mapping SOFT components to design decisions

| SOFT component | Literature concept it corresponds to | What should be compared next |
|---|---|---|
| Time-stepping layer | One-step and structural time integrators | Refinement studies for all five current methods, then BDF2/generalized-α only if a use case requires them |
| Nonlinear solve layer | Matrix-free L-BFGS, Armijo globalization, fallback/watchdog/rescue | Projected Newton, Project-on-Demand Newton, and barrier/contact line searches |
| Mechanical state and residual layer | Force/residual assembly and NumPy/JAX execution | Separate mechanical residual, contact residual, collision broad phase, and backend-independent diagnostics |
| Constitutive layer | StVK and stable Neo-Hookean constitutive kernels | Constitutive plug-ins with stress/tangent/energy verification and parameterized near-incompressible tests |
| Mesh and geometry layer | Structured tetrahedral generation and surface extraction | Mesh quality metrics, curved/high-order elements, remeshing, and cut-aware topology/enrichment |
| Not yet present | Collision, contact, friction, cutting | Start with surface extraction plus broad/narrow phase; select barrier, complementarity, or XPBD-style response as separate policies |

## 6. Focused literature case studies

The review separates collision detection from contact resolution because the
literature repeatedly shows that they have different performance and
correctness bottlenecks. A practical extension should have at least these
interfaces:

1. **Geometry representation:** simulation volume mesh, extracted surface
   triangles, optional collision proxy, thickness, and material/contact labels.
2. **Broad phase:** BVH/AABB, spatial hash, or grid producing candidate pairs.
3. **Narrow phase:** discrete distance/intersection and, when needed,
   continuous swept tests with closest-point witnesses.
4. **Contact state:** persistent pair IDs, normal/tangent frames, friction
   history, and barrier or compliance state.
5. **Response policy:** penalty/repulsion, position projection, complementarity,
   augmented Lagrangian, or incremental potential contact.
6. **Nonlinear coupling:** either monolithic residual/Jacobian coupling with
   the material solve or an explicitly documented staggered/contact projection
   loop.

The recommended first extension for a mechanics-first SOFT is a surface
BVH plus continuous distance queries feeding an incremental-potential or
barrier response. This follows IPC's central design advantage: non-penetration
is enforced by the potential and line search rather than by choosing an
arbitrarily large penalty stiffness. A lighter interactive branch can then
reuse the same candidate/contact representation with XPBD or augmented-
Lagrangian updates. A pure penalty implementation is useful as a diagnostic
baseline, but should not be the only production architecture because its
stiffness parameter couples contact accuracy to timestep and nonlinear
conditioning.

Cutting should not be treated as “collision with deletion.” Mesh surgery,
remeshing, element duplication, and XFEM enrichment change connectivity or
the approximation space. The Andrews/Ton-That/Kry work is particularly
relevant: XFEM and robust Booleans decouple cut geometry from the background
mesh and localize the added integration cost. That approach is a better future
extension point for SOFT than embedding cut-specific topology edits inside
the current tetrahedral force kernel.

### Focused case study: Dynamic Deformables and HOBAK

The [Dynamic Deformables: Implementation and Production Practicalities course](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/)
by Theodore Kim and David Eberle is especially relevant to SOFT because it
connects the papers to a working simulator and makes the implementation
choices inspectable through the open-source [HOBAK library](https://github.com/theodorekim/HOBAKv1).
It is not just another material paper: it is a systems-level reference for
force gradients, element eigendecompositions, implicit integration, assembly,
solver performance, collision detection, collision response, and production
failure modes.

The course also makes a useful distinction between illustrative algorithms and
production choices. The notes include Newmark and BDF1/BDF2 timestepper code
that performs multiple Newton--Raphson iterations, while explicitly describing
those versions as illustrative/research code rather than the production
method. This is directly relevant to the current SOFT work: adding a named
timestepper is not the same as matching the nonlinear solve, line search,
contact policy, warm starts, and engineering safeguards that make a production
solver usable.

| Dynamic Deformables / HOBAK topic | Relevance to SOFT |
|---|---|
| Deformation gradient and force-gradient construction | Confirms that constitutive and geometric derivatives deserve first-class APIs; it is a stronger path toward Newton or projected-Newton solvers than treating the L-BFGS residual as the only solver interface |
| Element Hessian eigendecomposition and positive-semidefinite filtering | Directly connects to stable Neo-Hookean robustness, projected Newton, and the missing Hessian-based solver comparison in SOFT |
| Newmark, BDF1, and BDF2 examples | Confirms that Newmark and BDF2 belong in the field coverage; also shows that timestep formulas must be evaluated together with Newton convergence and line search |
| Tetrahedral FEM production examples | Provides a high-value comparison point for the current linear-tet/stable-Neo-Hookean reference kernel, including large meshes and performance-oriented assembly |
| Proximity queries, continuous collision detection, and global intersection analysis | Supports the proposed separation of broad phase, narrow phase, persistent contact state, CCD, and global intersection recovery |
| Collision response for cloth, solids, and two-way coupling | Shows that contact robustness is a major subsystem rather than a small force term appended to the material model |
| Open-source code, regression tests, and production practicalities | Provides an implementation reference for solver diagnostics, scene-level tests, and failure handling that is closer to SOFT's needs than a paper-only survey |

The course also reports a production history: improved cloth collision and
response for *Coco*, three-dimensional solids for *Cars 3*, and two-way cloth-
body coupling for *Onward*. These claims should be read as production-system
evidence, not as controlled benchmark comparisons, but they establish the
regime in which robust collision and coupling are judged: visual plausibility,
failure resistance, and throughput across many shots, not only timestep
convergence on an unconstrained cantilever.

### Focused case study: Taku Komura's work

A scan of [Taku Komura's publication list](https://i.cs.hku.hk/~taku/publication.html)
shows several works that should be added to the review. They do not all belong
in the same solver category. The 2022 ARAP paper is directly relevant to the
current SOFT design: it replaces explicit rotation handling in an isotropic
ARAP/corotational energy with Cauchy--Green invariants and derives closed-form
derivatives and a Hessian suitable for Newton-type implicit solves. The paper
also discusses positive-semidefinite Hessian treatment. This makes it a useful
comparison point for StVK, stable Neo-Hookean, and the current L-BFGS residual
solver, even though it does not imply that ARAP is a universally better
material model.

The two collision papers are relevant to a future contact subsystem rather than
to the present material or timestepper implementation. They reinforce an
architectural separation between (1) broad-phase hierarchy construction and
traversal, (2) narrow-phase distance or intersection queries, and (3) contact
response. The binary-tree paper addresses compact BVH representation and
memory traffic; the GPU paper addresses parallel traversal and workload
divergence. Neither paper supplies a contact law or a nonlinear mechanics
solve, so they should not be counted as evidence that the current implicit
stepper is missing a globalization strategy.

The XFEM and material-point papers are also important omissions, but they are
topology-change and fracture references. The XFEM work decouples an explicit
crack surface from the background deformation mesh, while the material-point
work combines MPM, local continuum damage mechanics, Voronoi crack surfaces,
and rigid-fragment conversion. These approaches are strong candidates for a
future cutting/fracture branch, but they should not be folded into the first
linear-tetrahedral FEM/contact path merely to make the literature list appear
complete.

| Komura work | What it adds to the design review | Recommended status for SOFT |
|---|---|---|
| Isotropic ARAP using Cauchy--Green invariants | A constitutive-energy alternative with analytic gradient/Hessian information and a Newton-compatible PSD treatment | Add now as a material/second-order-solver benchmark; retain StVK and stable Neo-Hookean as existing baselines |
| Binary Ostensibly-Implicit Trees | A compact BVH representation for broad-phase collision detection | Future collision extension; first implement a clear CPU broad-phase interface before optimizing representation |
| Bulk-synchronous parallel BVH traversal | A GPU strategy for simultaneous traversal of dynamic BVHs with reduced divergence | Future GPU broad-phase extension; benchmark only after narrow phase and contact state are separated |
| Displacement-Correlated XFEM | Remeshing-free crack representation with explicit cut-surface geometry | Future cutting/fracture extension; preserve the background mesh and add enrichment locally |
| Brittle fracture with material points | An MPM/CDM route to damage, crack surfaces, and rigid fragments without ordinary FEM remeshing | Separate fracture solver family; do not use it as the baseline contact implementation |

Komura's older musculoskeletal and muscle-model papers are useful historical
biomechanics references, but they do not fill a missing requirement in the
current first-order tetrahedral FEM review. They can be added in a dedicated
biomechanics subsection if SOFT later targets anatomically structured muscle
or articulated soft tissue.

### Focused case study: Matthias Teschner and Matthias Müller

The earlier version of this review covered their major survey and method
families, but not the complete set of papers that informs the present SOFT
design. The added works above close the most important gaps:

- Teschner's work now covers the collision pipeline from broad phase
  (optimized spatial hashing), through geometric and penetration queries,
  point-based contact, complex deformable solids, and real-time tetrahedral
  cutting. These papers are particularly important because they show that
  collision detection, contact response, and cutting are separate design
  problems rather than extensions of the material kernel.
- Müller's work now covers the interactive alternatives around FEM: meshless
  shape matching, hierarchical PBD, oriented particles, strain-based dynamics,
  physically based shape matching, and auxiliary collision geometry. It also
  covers the much more directly relevant contact-solver branch through
  non-smooth Newton methods for coupled hyperelastic and rigid-body dynamics.
- The review intentionally does not list every paper by either author. Fluid,
  hair, rendering, animation-authoring, and unrelated rigid-body papers are
  outside the present scope. The selection is intended to cover methods that
  affect SOFT's discretization, constitutive modeling, timestep/solver
  design, collision detection, contact response, or topology change.

The most consequential omission for the current implementation was the
non-smooth Newton paper: unlike PBD and shape matching, it connects
hyperelastic tetrahedral FEM, frictional contact, rigid/deformable coupling,
and an implicit nonlinear solve in one framework. It should be treated as a
future contact-solver benchmark alongside IPC and projected-Newton methods,
not as evidence that the current contact-free L-BFGS implementation is
incorrect.

## 7. Thematic method families

The families below are deliberately thematic rather than chronological. The
publication dates remain in the citations so that the evolution of the field
can still be reconstructed, but methods are grouped by the design mechanism
that matters when choosing or extending a solver.

### Early continuum and physically based deformable models

The 1987--1988 Terzopoulos line established a graphics vocabulary in which
curves, surfaces, and solids are discretized mechanical models driven by
elastic or inelastic differential equations. The representation is close to
continuum mechanics, but the goal is animation and shape modeling rather than
certified engineering prediction. Materials include elastic, viscoelastic,
plastic, and fracture-like behavior, often expressed through evolving
reference configurations and force laws.

The important legacy is architectural: state, constitutive behavior, external
forces, constraints, and obstacle interaction are separate concepts. The
papers also make clear that collision/contact is not solved merely by choosing
a better integrator; it needs its own geometric and constraint machinery.

### Implicit large-step mechanics

Baraff--Witkin and later implicit FEM/reduced-model work use a mechanics-first
strategy: choose an implicit method, linearize or optimize the step, and solve
the resulting system. The main gain is timestep robustness for stiff stretch,
bending, or volumetric elasticity. Backward Euler is especially attractive
because it tolerates large steps and damps unresolved motion, although its
numerical damping can reduce oscillatory accuracy.

Typical representations are triangle cloth, tetrahedral or subdivision FEM,
and reduced modal/hierarchical coordinates. Typical solvers are modified
conjugate gradient, Newton or quasi-Newton iterations, prefactorized global
steps, or constrained optimization. The interactive regime is obtained by
reducing degrees of freedom, exploiting sparsity, warm-starting, accepting
inexact convergence, and using a fixed compute budget.

### Robust cloth collision and contact

Cloth papers emphasize that collision detection can dominate the cost even
when internal dynamics are inexpensive. A typical pipeline is:

1. broad phase using spatial coherence, bounding boxes, or a hierarchy;
2. narrow phase on vertex--face and edge--edge pairs;
3. thickness-aware proximity or continuous collision tests;
4. friction/contact response through repulsion, impulses, velocity filtering,
   or constraints.

Bridson et al. are representative of a hybrid approach: use inexpensive
repulsion for most resting interactions and a more robust geometric mechanism
for the few collisions that remain. This pattern is attractive for animation
because it reserves expensive contact work for difficult events.

Selle et al. show the complementary production strategy for very dense cloth:
history-based collision state avoids rediscovering the same contacts, while
parallel time evolution and collision processing make meshes with millions of
triangles practical. The result is an important IEEE TVCG reference for
high-resolution cloth, but its target is dependable hero-quality simulation,
not the bounded latency expected from an interactive manipulation tool.

### Constraint and position-based simulation

PBD and XPBD move the primary solve from forces and velocities to positions
and constraints. Stretch, bending, volume, attachment, and collision are
handled in a common iterative projection framework. This makes the solver easy
to bound by a fixed number of iterations and naturally prevents or removes
penetration. XPBD adds compliance so stiffness is less directly tied to the
number of iterations and timestep.

The trade-off is that a visually stable result is not automatically the same
as a high-accuracy solution of the original force-based dynamics. Constraint
ordering, iteration count, compliance, collision refresh, and velocity
reconstruction affect damping, convergence, and physical fidelity.

### Reduced and hierarchical representations

Reduced coordinates, modal bases, volumetric subdivision, deformation textures,
and adaptively sampled surface points reduce the state that must be integrated
or queried. They are particularly effective for interactive manipulation and
haptics, where the geometry used for rendering or collision can be much richer
than the dynamics state.

The limitation is model coverage: a reduced basis or embedded texture is most
effective for deformation families represented in its construction. Large
topological changes, cutting, fracture, arbitrary contact, or strongly
nonlinear material changes can invalidate the reduction or require local
refinement.

### Optimization and local-update implicit solvers

Projective Dynamics, MPRGP-style constrained optimization, VBD, and AVBD all
recast some form of implicit dynamics as an optimization or local-update
problem. Their common graphics insight is that a solver can be useful under a
fixed iteration budget if every partial iteration remains bounded and visually
reasonable. Prefactorization, local projections, block updates, coloring,
warm-starts, and augmented Lagrangians reduce the cost of each iteration or
improve robustness around contacts.

These methods are close in spirit to the current implicit L-BFGS prototype,
but they differ in what is approximated: L-BFGS approximates an inverse
Jacobian action, Projective Dynamics separates local and global energy steps,
and VBD uses local vertex block descent on the implicit-Euler variational
energy.

## 8. Collision detection and contact methods

### Collision detection

Across the reviewed literature, the main representations are:

- **Bounding-volume hierarchies:** AABB, OBB, sphere trees, and deformable BVH
  updates exploit spatial coherence and are common for triangle meshes.
- **Spatial partitioning:** uniform grids, spatial hashing, voxelization, and
  subdivision cells reduce candidate pairs for large or changing scenes.
- **Distance fields and proximity queries:** signed-distance grids and dynamic
  deformation textures provide fast distance or penetration queries, often at
  the cost of preprocessing or resolution dependence.
- **Continuous collision detection:** swept primitives, interval methods, and
  conservative advancement reduce temporal aliasing when objects can pass
  through one another in one step.
- **Reduced or sampled geometry:** pointshells, modal samples, and collision
  proxies decouple collision cost from render-mesh complexity.

Most systems use discrete detection every simulation step and add continuous
tests, conservative advancement, or substepping only for fast or thin objects.

### Contact resolution

The main response families are:

| Response family | Typical benefit | Typical cost or limitation |
|---|---|---|
| Penalty/repulsion forces | Simple, modular, easy to combine with existing forces | Requires stiffness tuning and can introduce timestep restrictions or jitter |
| Impulses/velocity filtering | Good for impacts and fast separation | Resting contact and friction require stabilization or persistent contact state |
| Position projection/PBD | Very robust under a fixed iteration budget; direct penetration removal | Physical compliance, damping, and convergence depend on projection iterations |
| Lagrange multipliers / complementarity | Direct non-penetration and principled friction/contact constraints | Indefinite or mixed systems and changing contact sets can be expensive |
| Augmented Lagrangian | Handles hard constraints while improving conditioning relative to extreme penalties | Adds dual variables and update parameters |
| Hybrid methods | Allocate expensive robust treatment only to difficult contacts | More implementation complexity and mode-switching decisions |

## 9. Material parameter practice

The reviewed works use several recurring material abstractions:

- mass-spring or particle systems for cloth, hair, and highly interactive
  prototypes;
- continuum-inspired cloth energies for stretch, shear, bending, and damping;
- linear elasticity and Saint Venant--Kirchhoff-like finite elements for
  volumetric solids;
- corotational or hyperelastic FEM for larger rotations and nonlinear solids;
- constraint/compliance parameters in PBD and XPBD;
- reduced modal or hierarchical bases for restricted deformation families;
- inelastic, viscoelastic, plastic, and fracture extensions for specialized
  animation effects.

Exact material settings are inconsistently reported. Common practice is to
choose density, stiffness, Poisson ratio or constraint compliance, and damping
to achieve a desired qualitative response, then tune contact thickness and
friction separately. This is appropriate for animation but means that a
paper's “interactive” result is not automatically a physically validated
material simulation.

For the current SOFT soft-body prototype, this suggests separating three
claims: constitutive correctness, trajectory accuracy, and interactive
robustness. A method can satisfy the third through numerical damping or
bounded inexact solves while failing the first two for a scientific undamped
test.

## 10. Interactive regimes

The literature contains at least four regimes:

| Regime | Practical requirement | Typical methods |
|---|---|---|
| Offline animation | High visual quality; frame time may be minutes or longer | Early continuum models, high-resolution cloth, detailed contact |
| Interactive graphics | User feedback around frame rate; bounded latency is valuable | PBD/XPBD, reduced models, Projective Dynamics, optimized implicit solvers |
| Real-time control or robotics | Stable and repeatable updates, often tens to hundreds of Hz | Reduced FEM, SOFA-style implicit FEM, constraint-based solvers |
| Haptic interaction | Contact/force loop commonly much faster than graphics display | Reduced models, sampled pointshells, local collision/contact updates |

The same paper can occupy multiple regimes: a reduced model may be interactive
while a full-resolution reference run is offline. Reported frame rates also
usually exclude preprocessing, asset loading, rendering, or collision costs;
comparisons should therefore use the complete application loop whenever
possible.

## 11. Framework, library, and author case studies

The next sections are grouped by type of evidence. Frameworks and libraries
come first, followed by researcher/algorithm case studies and then other
open-source engines. This prevents implementation architecture, constitutive
models, and historical author contributions from being mixed into one
chronological list.

### Open-source frameworks

### SOFA

[SOFA](https://www.sofa-framework.org/) is an open-source framework
targeted at interactive multiphysics, especially medical simulation and
robotics. Its [collision documentation](https://sofa-framework.github.io/doc/simulation-principles/multi-model-representation/collision/)
describes a modular pipeline with broad phase, narrow phase, contact output,
and selectable penalty, persistent, or constraint-based responses. Its
[simulation tutorial](https://sofa-framework.github.io/doc/simulation-principles/example-simple-body/)
shows the separation between integration scheme, linear solver, mechanical
model, and collision response. SOFA is valuable here because it demonstrates
how a production research framework exposes solver and contact choices as
components rather than baking them into one deformable-body algorithm.

The architectural reference is [Faure et al., “SOFA, a Multi-Model Framework
for Interactive Physical Simulation”](https://www.lirmm.fr/~gilles/papers/faure_springer12.pdf).
SOFA organizes independently developed components in a scene graph and uses
multi-model representations and mappings so that mechanical, collision, and
visual representations of the same object need not share the same mesh. Its
component boundaries include degrees of freedom, force fields and constraints,
differential equations, time integration, linear solvers, collision detection,
and contact handling. This directly supports the proposed SOFT separation of
the constitutive/mechanical kernel, timestepper, nonlinear solver, collision
pipeline, and contact policy.

SOFA is therefore more than another FEM implementation in this review. It is
an architectural comparator for extensibility and algorithm comparison. The
current SOFT code is intentionally much smaller and currently contact-free,
but SOFA provides a useful target for future interfaces without requiring
SOFT to adopt SOFA's full scene-graph or plugin system.

The [SOFA source repository](https://github.com/sofa-framework/sofa) is also
listed separately because it is the implementation artifact, while the SOFA
website and architecture paper document the framework's intended component
boundaries and use cases.

### VegaFEM

[VegaFEM](https://github.com/jjcao/VegaFEM) is an open-source research library
for physically based simulation, including interactive deformable simulation
utilities. Its documented examples include Saint Venant--Kirchhoff,
corotational, linear, mass-spring, and invertible FEM variants, and its
interactive simulator includes implicit Newmark among its choices. VegaFEM is
useful as a reference for the engineering trade-off between an explicit method
with more numerical damping and an implicit method with a more expensive solve.
Collision handling is not a single universal VegaFEM feature; applications
often connect an external collision system or specialized contact code.

### HOBAK

[HOBAK: A Library for Squashing Things](https://github.com/theodorekim/HOBAKv1)
is an open-source companion implementation for Theodore Kim's deformable-body
work and the [Dynamic Deformables course](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/).
It belongs in this list because it exposes working code for tetrahedral FEM,
hyperelastic constitutive models, deformation-force derivatives, element
Hessian processing, implicit integration, collision detection, and collision
response. It is therefore a closer implementation reference for the current
SOFT prototype than a paper-only entry.

HOBAK should not be interpreted as a drop-in general-purpose multiphysics
framework like SOFA, nor as a benchmark with the same API or acceptance tests
as SOFT. Its value is different: it shows how production-oriented graphics
simulation connects constitutive kernels, Newton iterations, eigensystem-based
Hessian safeguards, timestep control, contact handling, and regression scenes.
The review consequently uses HOBAK both as an open-source system reference and
as implementation material for the Dynamic Deformables discussion above.

### Researcher and algorithm case studies

#### Doug James: reduced dynamics, collision processing, and haptics

Doug James's work was missing as a connected thread in the earlier version of
this review. It is important because it explains a major route by which
graphics systems obtain interactive deformable simulation: reduce the dynamic
state, exploit temporal or geometric coherence, and use a collision/contact
representation whose cost is controlled independently of the full mesh.

The early [ArtDefo](https://graphics.stanford.edu/~djames/publication/artdefo-accurate-real-time-deformable-objects/)
paper uses a boundary-element formulation for static linear elasticity and
accelerates updates when only a few boundary conditions change. This is not a
tetrahedral FEM timestepper, but it is directly relevant to the distinction
between a mechanics formulation and the runtime policy used to make interaction
possible.

[Precomputing Interactive Dynamic Deformable Scenes](https://graphics.stanford.edu/~djames/publication/precomputing-interactive-dynamic-deformable-scenes/)
goes further by tabulating nonlinear deformation dynamics, impulse responses,
and self-contact offline. The runtime system then evaluates a low-dimensional
model, so contact is effectively compiled into the admissible scene response.
This is a powerful graphics answer to interaction, but it applies only to
repeatable interaction families and should not be confused with general-purpose
online simulation.

The [BD-Tree](https://graphics.cs.cmu.edu/projects/stvk/) work makes collision
detection for reduced deformable models output-sensitive. Its bounded
deformation hierarchy is relevant to a future SOFT design in which the
mechanical mesh and collision representation are separate. It avoids requiring
the collision system to traverse the entire high-resolution deformable mesh at
every query.

[Real-Time Subspace Integration for St. Venant--Kirchhoff Deformable Models](https://graphics.cs.cmu.edu/projects/stvk/)
is the closest James-related comparison to the current SOFT mechanics. It
uses StVK elasticity, nonlinear geometric deformation, low-dimensional
subspaces, precomputed cubic reduced internal-force expressions, and implicit
Newmark integration. The reported kilohertz dynamics rate is achieved because
the online cost depends mainly on the reduced dimension, not the original
tetrahedral or surface resolution. This is the missing qualification when
comparing full-order implicit FEM against graphics interaction results: the
largest speedups often come from model reduction, not from choosing a more
aggressive full-order timestep.

The related [time-critical distributed contact](https://diglib.eg.org/bitstreams/a4e24227-0851-4a9b-ab3c-d55282124551/download)
and [six-DoF haptic contact](https://graphics.stanford.edu/~djames/research/)
work show the haptic regime explicitly. Point samples, signed-distance fields,
multiresolution contact, and graceful degradation are used to maintain a
kilohertz force-feedback loop. The objective is bounded latency and plausible
forces, not a tightly converged full-order trajectory at the display rate.

Finally, [Skipping Steps in Deformable Simulation with Online Model Reduction](https://doi.org/10.1145/1618452.1618469)
learns a reduced nonlinear model during the simulation and selectively replaces
full-order steps with reduced steps. This is especially relevant to the
scientific-versus-interactive distinction in SOFT: it exposes a throttle
between conservative full solves and fast approximate previews, rather than
pretending that one timestep and tolerance are optimal for all phases of a
motion.

#### Eftychios Sifakis: robust FEM flesh, hybrid representations, and cutting

Eftychios Sifakis is another major omission from the earlier review. His work
is central to the graphics FEM lineage that connects linear tetrahedral meshes,
implicit or quasistatic Newton solves, inversion robustness, character flesh,
embedded collision representations, and cutting.

[Robust Quasistatic Finite Elements and Flesh Simulation](https://graphics.cs.wisc.edu/Papers/2005/TSIF05/)
is especially relevant to the current solver investigation. It shows that
“implicit” does not eliminate all restrictions: element inversion, indefinite
stiffness matrices, large boundary-condition jumps, and discontinuous contact
forces can still constrain a solve. The method modifies element stiffnesses to
obtain a positive-definite global system suitable for conjugate gradients and
constructs smooth elastic forces around highly deformed or inverted elements.
This is directly related to the current SOFT diagnostics: a failed nonlinear
solve may reflect tangent definiteness and globalization, not merely an overly
strict residual tolerance.

[Hybrid Simulation of Deformable Solids](https://graphics.cs.wisc.edu/Papers/2007/SSIF07/)
is a key architectural reference. It embeds sample points in an initial mesh
and permits the elastic, collision, and constraint representations to differ.
This avoids forcing collision, plasticity, and fracture to use the same
topological representation as the constitutive FEM mesh. That design is a
strong argument for keeping future SOFT collision geometry and contact
resolution modular rather than embedding them directly into the current
tetrahedral constitutive kernel.

[Arbitrary Cutting of Deformable Tetrahedralized Objects](https://diglib.eg.org/server/api/core/bitstreams/cb0b56e2-3e56-4672-954a-f2aa92d510c9/content)
extends the same idea to progressive cuts. It is relevant to SOFT because
cutting exposes a limitation of a fixed-mesh scientific benchmark: after a cut,
the state space, topology, contact geometry, and admissible timestep policy
all change. Cutting should therefore be treated as a separate extension, not
as a small collision-response option.

The [Sifakis--Barbic FEM course](https://viterbi-web.usc.edu/~jbarbic/femdefo/)
is also an important review source rather than only a tutorial. It organizes
the field around deformation gradients, constitutive laws, tetrahedral
discretization, invertible elasticity, implicit linear solves, multigrid, and
model reduction. It provides the clearest external checklist for evaluating
whether SOFT has adequate coverage of the mechanics-first FEM pipeline.

The later [Differentiable Implicit Soft-Body Physics](https://arxiv.org/abs/2102.05791)
paper connects this lineage to differentiable simulation. It defines implicit
soft-body states through energy minimization and differentiates through the
implicit solve using matrix-free reverse-mode derivatives. This is relevant to
the JAX backend in SOFT: the current backend demonstrates differentiable
force/tangent operations, but it does not yet expose an adjoint or
gradient-through-time interface for inverse design or policy optimization.

#### Miguel Otaduy: implicit contact, friction, haptics, and material fitting

Miguel Otaduy was underrepresented in the earlier review. His work is central
to the contact side of the field and complements the mechanics-first work of
Sifakis, James, Kim, and others.

[Implicit Contact Handling for Deformable Objects](https://media.disneyanimation.com/uploads/production/publication_asset/32/asset/EG2009_implicit_contact.pdf)
is a key reference for a future SOFT contact system. It formulates contact
with internal dynamics, non-penetration inequalities, complementarity, and an
iterative constraint-anticipation solver. The target behaviors—stable stacking,
rolling, sliding, impacts, and heterogeneous deformable coupling—are exactly
the behaviors that an energy-envelope test cannot establish by itself.

[Efficient Simulation of Contact between Rigid and Deformable Objects](https://www.researchgate.net/publication/228448338_EFFICIENT_SIMULATION_OF_CONTACT_BETWEEN_RIGID_AND_DEFORMABLE_OBJECTS)
addresses the dense coupling that appears when a rigid body contacts a large
deformable region. Its partitioning of rigid-only, deformable-only, and mixed
constraints is a useful design reference for keeping future contact solves
structured instead of treating every contact constraint as an unstructured
global system.

Otaduy's haptics work, including [Haptic Rendering of Objects with Rigid and
Deformable Parts](https://www.sciencedirect.com/science/article/abs/pii/S0097849310001350),
shows a different acceptance regime from the scientific autotuner. Force
feedback requires bounded latency, stable contact forces, and graceful
degradation. A trajectory can be physically approximate yet operationally
successful if the contact force remains stable and responsive.

Finally, [Modeling and Estimation of Energy-Based Hyperelastic Objects](https://onlinelibrary.wiley.com/doi/10.1111/cgf.12840)
is relevant to material validation. It treats the energy model and its
parameters as quantities to be inferred from deformation data. This reinforces
that a solver can be numerically converged while the material model is still
poorly calibrated; SOFT should eventually separate solver verification from
material-identification validation.

#### Paul G. Kry: reduction, embedded models, and adaptive interaction

Paul G. Kry was also underrepresented. His work is important because it
connects deformable simulation to the graphics-side questions of how much of a
full model must remain dynamic, how visual resolution can be separated from
mechanical resolution, and how collision queries can be accelerated for
reduced models.

[EigenSkin](https://www.cs.mcgill.ca/~kry/pubs/eigenskin/) uses data-dependent
eigenbases to render large nonlinear FEM character deformations in hardware.
It is primarily a quasistatic, pose-driven representation rather than an
online dynamic timestepper, but it is a clear example of moving expensive
simulation offline and retaining only a compact runtime deformation model.

[FastLSM](https://doi.org/10.1145/1276377.1276480) uses embedded geometry and
lattice shape matching to obtain robust, linear-time large-deformation
updates. The paper explicitly prioritizes robustness and interactive speed over
strict physical consistency, making it a useful comparator for the interactive
autotuner rather than the scientific FEM autotuner.

[Preserving Topology and Elasticity for Embedded Deformable Models](https://www.cs.mcgill.ca/~cg/projects/composite/)
shows how a coarse embedded model can preserve disconnected topology,
heterogeneous materials, and empty space while driving detailed geometry. This
is directly relevant to a future SOFT architecture with distinct mechanical,
visual, and collision representations.

Kry's recent work also reaches adaptive rigidification and learned collision
queries. [Adaptive Rigidification of Elastic Solids](https://doi.org/10.1145/3528223.3530156)
reduces runtime work by treating regions that are not currently deforming as
rigid. [Active Learning Neural C-space Signed Distance Fields](https://www.cs.mcgill.ca/~kry/)
learns reduced-model self-collision queries from actively selected samples.
These are not replacements for a verified full-order solver, but they identify
two concrete paths to interactive performance: adaptive dynamic resolution and
reduced collision models.

### Other open-source engines and implementation material

#### Tutorials and implementation material

The [Baraff and Witkin paper](https://www.cs.cmu.edu/~baraff/papers/sig98.pdf),
the [PBD paper](https://doi.org/10.2312/PE/vriphys06/071-080), the
[XPBD paper](https://matthias-research.github.io/pages/publications/XPBD.pdf),
and the [Projective Dynamics paper](https://www.projectivedynamics.org/projectivedynamics.pdf)
are unusually useful implementation references because they expose the
algorithmic structure rather than only reporting application results.

#### NVIDIA Warp

[NVIDIA Warp](https://nvidia.github.io/warp/) is an open-source Python and
kernel-programming framework for GPU-accelerated simulation, robotics, and
machine learning. It is not a single prescribed soft-body method. Its relevant
building blocks include differentiable kernels, sparse linear algebra, BVHs,
hash grids, and a finite-element toolkit. The examples and projects built on
Warp cover FEM, contact, optimization, and MPM, so Warp is best classified as
an execution and algorithm-prototyping platform rather than as a direct
alternative to SOFT's tetrahedral solver.

For this review, Warp matters in three ways: it provides a route to a GPU
backend for element and residual kernels; it makes reverse-mode differentiation
of simulation programs practical; and it supports independent experiments
with FEM, contact, and MPM without forcing those methods into one monolithic
engine. The trade-off is that a Warp implementation still has to choose its
constitutive model, timestepper, nonlinear solver, collision representation,
and contact policy. Warp therefore does not by itself validate a particular
solver choice or acceptance criterion used by SOFT.

#### Taichi and differentiable MPM systems

[Taichi Lang](https://github.com/taichi-dev/taichi) is an open-source
data-oriented programming language and runtime, not a soft-body solver in the
same sense as SOFA or PhysX. It is nevertheless highly relevant to the
graphics soft-body literature because it enabled compact, high-performance
implementations of MLS-MPM, differentiable MPM, and soft-robot simulation.
The related [Taichi MPM implementation](https://github.com/yuanming-hu/taichi_mpm)
is especially relevant for large deformation, contact, displacement
discontinuity, and two-way rigid-body coupling.

The Taichi lineage is a different branch from SOFT: it uses particles and a
background grid, handles topology changes and extreme deformation naturally,
and is attractive for differentiable control and design. It does not provide
the same first-order tetrahedral FEM convergence study as the scientific
SOFT autotuner. Its results should therefore be compared as a separate MPM
solver family, not mixed into FEM timestep rankings.

#### Bullet Physics

[Bullet](https://github.com/bulletphysics/bullet3) does include deformable
objects through `btSoftBody`; the [Bullet user manual](https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/Bullet_User_Manual.pdf)
describes cloth and volumetric soft-body dynamics, and the API documents
`btSoftBody` as supporting both cloth and volumetric bodies. The important
qualification is that Bullet's soft-body subsystem is not a modern
finite-element benchmark comparable to SOFT, HOBAK, or PhysX FEM. It is a
graphics/physics-engine subsystem based on particle, link, face, cluster, and
position/velocity-solver concepts, with engine-oriented collision and response
policies.

Bullet is still relevant as a practical baseline for interactive contact and
engine integration. It should not be used as evidence for the accuracy of
StVK or stable Neo-Hookean tetrahedral FEM unless the exact soft-body model,
solver settings, and validation tests are specified.

#### NVIDIA PhysX and Isaac Sim

[NVIDIA PhysX 5](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/DeformableBodyOverview.html)
does support deformable surfaces and deformable volumes using FEM. Its current
documentation describes separate simulation and collision representations,
GPU-only deformable-body execution, FEM materials, and GPU solver choices; it
recommends TGS for deformable-body simulation. This makes PhysX a relevant
production-engine reference for mesh separation, GPU execution, and contact
integration, although its implementation details and acceptance tests are not
the same as the research-oriented SOFT examples.

[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) is an application and
robotics-simulation environment built around PhysX, not an independent
deformable-body algorithm. Its deformable-object tutorials expose PhysX FEM
soft bodies using separate tetrahedral simulation and collision meshes. Isaac
Sim is therefore relevant for the interactive robotics regime, sensor/control
integration, and GPU deployment, but its soft-body solver references should be
attributed to PhysX rather than counted as a separate timestepper family.

#### MuJoCo

[MuJoCo](https://mujoco.readthedocs.io/en/3.3.3/modeling.html) is historically
a rigid multibody and model-based-control engine. Older composite mechanisms
can create objects that look soft, but they are not the same as a continuum
deformable-body discretization. MuJoCo 3.0 introduced `flex`, a native
deformable-object model supporting one-, two-, and three-dimensional elements,
including tetrahedral volumes.

MuJoCo is therefore relevant to SOFT mainly as a robotics/control reference
and as evidence that a production engine may combine rigid multibody dynamics,
soft contact, and newer deformable elements in one API. Its principal regime is
control and reinforcement learning with bounded simulation cost. It is not a
direct validation reference for the nonlinear tetrahedral FEM and trajectory
accuracy tests in the scientific SOFT autotuner.

#### SuperDex / Mochi

[SuperDex](https://projectsuperdex.com/) is a particularly close systems
reference for the future direction of SOFT: it is a contact-first engine for
dexterous manipulation, with rigid bodies, tetrahedral soft volumes, and
experimental shells and rods. Its public documentation describes an implicit
time integrator, quasi-Newton iterations with line search, positive-semidefinite
Jacobian/tangent projections, and multiple linear-solver choices. Its soft
materials include linear elastic, Neo-Hookean, StVK, ARAP, and active variants.

The contact architecture is also important. SuperDex uses compliant,
differentiable contact forces; signed-distance fields represent collider
geometry, while quadrature samples represent the colliding deformable body.
This is a different design from the current contact-free SOFT prototype,
but it provides a concrete example of how constitutive forces, implicit solves,
collision geometry, and contact response can be composed for manipulation.

The [open-source SuperDex repository](https://github.com/facebookresearch/project_superdex)
currently provides a 2026 technical/citation artifact rather than a mature
peer-reviewed benchmark paper. Its documentation is valuable for architectural
comparison, but qualitative claims about stability or performance should not be
treated as independent validation until reproducible benchmark data are
available.

#### Drake

[Drake](https://drake.mit.edu/) is an open-source robotics and control framework
whose `MultibodyPlant` includes FEM-based deformable bodies. Its
[deformable-body API](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_deformable_body.html)
connects tetrahedral meshes, physical properties, constitutive parameters,
boundary conditions, and external forces to the multibody scene. The FEM layer
exposes deformation gradients, residuals, and tangent matrices, and supports
automatic differentiation for optimization and control workflows.

Drake is also relevant to the future contact design: its deformable-contact
work includes a convex frictional rigid--deformable formulation with a
corotational material and a positive-semidefinite Hessian. The default FEM
configuration is intended as a hard-rubber-like starting point rather than a
universal material calibration. Drake should consequently be classified as an
engineering and robotics/control reference, not as a graphics production
engine or a direct replacement for the scientific SOFT verification suite.

### Capability comparison with SOFT

The following matrix compares the current contents of SOFT with the open
source systems reviewed above. “Separate” means that the system provides a
distinct representation or module; “configuration-dependent” means that the
system contains the building blocks but does not prescribe one universal
choice. A dash is intentional: the feature is outside the system's primary
scope, not necessarily an omission.

| System | Deformable representation and materials | Time stepping and nonlinear solve | Collision and contact | Backends and intended regime | What SOFT has or does not yet have |
|---|---|---|---|---|---|
| **SOFT SOFT** | First-order tetrahedral FEM; StVK and stable Neo-Hookean; lumped mass | Semi-implicit Euler; implicit backward Euler, midpoint, trapezoidal, and Newmark; matrix-free L-BFGS with globalization/fallback diagnostics | Jacobian-feasibility checks only; no collision detection or contact response | NumPy and JAX; scientific trajectory/energy studies plus separate interactive autotuning | Small, inspectable mechanics-first reference; currently lacks contact, cutting, reduced models, GPU-native kernels, and a modular scene graph |
| SOFA | Multiple mechanical models, including FEM, springs, constraints, and mappings; material/force-field choice is component-dependent | Pluggable integrators, linear solvers, nonlinear policies, and simulation-loop components | Modular broad phase, narrow phase, penalty, persistent, constraint, friction, and contact components | CPU/GPU components and interactive medical, haptic, and robotics research | SOFT has a much smaller single-model scope; SOFA is the architectural reference for future modular collision/contact interfaces |
| HOBAK | Tetrahedral FEM, hyperelastic materials, inversion-safe Hessians, and production-oriented solid examples | Implicit integration and Newton-style nonlinear solves with Hessian processing and practical safeguards | Collision detection and response are included for graphics scenes | Primarily CPU research/graphics implementation material | SOFT has a cleaner multi-timestepper verification harness and NumPy/JAX portability; HOBAK has the more complete graphics contact pipeline |
| VegaFEM | Linear, corotational, StVK, mass-spring, and invertible FEM variants | Explicit and implicit methods, including implicit Newmark; library/application dependent | Application-dependent or externally connected collision/contact | CPU research and interactive deformable simulation | SOFT covers more modern timestepper comparison and L-BFGS diagnostics; VegaFEM covers a broader established FEM model collection |
| PolyFEM | High-order triangles/tetrahedra, polygons/polyhedra, splines, and multiple constitutive models | Configuration-dependent transient integration and nonlinear/linear solver stack | Problem-dependent boundary and contact handling | CPU research/engineering and accuracy-oriented prototyping | SOFT is deliberately limited to linear tetrahedra; PolyFEM is the reference for higher-order elements, richer geometry, and accuracy-per-element extensions |
| NVIDIA Warp | FEM toolkit plus MPM and user-defined particle/grid methods; constitutive model is program-defined | User-programmed kernels and sparse solvers; no single prescribed timestepper or nonlinear method | BVHs, grids, and application-defined contact pipelines | GPU/CPU kernel programming, differentiable simulation, robotics, and optimization | SOFT has a prescribed scientific solver experiment; Warp offers the execution substrate for future GPU and autodiff implementations |
| Taichi / Taichi MPM | MLS-MPM particles and grid; soft robotics, fracture, and large-deformation applications | Program-defined particle/grid update and differentiable simulation pipelines | Application-specific particle/grid and rigid coupling | CPU/GPU data-oriented runtime; graphics, robotics, inverse design | SOFT is FEM rather than MPM; Taichi is a separate solver family for extreme deformation and topology change |
| Bullet `btSoftBody` | Particle/link/face/cluster soft bodies; engine-oriented material parameters | Engine soft-body updates and iterative constraint/response policies | Integrated interactive collision and response | CPU general physics engine; games and robotics prototypes | SOFT has continuum FEM constitutive verification; Bullet is a practical interactive contact baseline, not a like-for-like FEM comparison |
| NVIDIA PhysX 5 | FEM deformable surfaces and volumes with separate simulation/collision meshes | Production solver choices and GPU-oriented deformable-body integration | Integrated collision/contact with distinct collision representation | GPU-focused production engine; Isaac and interactive robotics/graphics | SOFT exposes solver mathematics more transparently; PhysX demonstrates the production value of separate collision meshes and GPU execution |
| Isaac Sim | PhysX deformable objects exposed in a robotics/sensor/control environment | Inherited from PhysX and configured through the application | PhysX collision/contact plus robotics scene integration | GPU robotics simulation and interactive deployment | Not an independent timestepper family; useful for evaluating SOFT-like methods in a complete robotics loop |
| MuJoCo | Rigid multibody core plus native `flex` 1D/2D/3D deformable elements, including tetrahedra | Control-oriented dynamics and solver configuration; not a dedicated nonlinear FEM benchmark | Integrated soft contact and constraints | CPU/GPU-supported robotics/control and reinforcement learning workflows | SOFT is more explicit about nonlinear FEM accuracy; MuJoCo is stronger as a bounded-cost control environment |
| SuperDex / Mochi | Tetrahedral soft volumes; experimental shells/rods; linear elastic, Neo-Hookean, StVK, ARAP, and active models | Implicit stages, quasi-Newton line search, PSD tangent projections, and multiple linear solvers | Contact-first compliant/differentiable SDF contact with quadrature samples | Contact-rich dexterous manipulation, RL, and interactive robotics | Closest architectural comparison for future SOFT contact work; SOFT currently has no contact or assembled solver stack |
| Drake | Tetrahedral FEM deformable bodies with configurable constitutive laws, boundary conditions, and physical properties | FEM residual/tangent formulation integrated with `MultibodyPlant`; autodiff and robotics solvers | Rigid--deformable contact and frictional contact formulations | CPU robotics/control and engineering workflows | SOFT is a focused timestepper/solver experiment; Drake provides the broader multibody, contact, and optimization context |

The matrix highlights an important asymmetry. SOFT currently compares
several time integrators and a matrix-free nonlinear solver more systematically
than most general-purpose engines expose publicly. Conversely, nearly all of
the mature interactive systems provide collision/contact, multiple geometric
representations, or application integration that SOFT does not yet provide.
The systems are therefore complementary references rather than a single
performance ranking.

| System | Soft/deformable support | What it contributes to the SOFT review | Classification |
|---|---|---|---|
| NVIDIA Warp | FEM toolkit, differentiable kernels, sparse solvers, geometry queries, and MPM projects | GPU execution, automatic differentiation, and a platform for implementing alternative FEM/MPM/contact solvers | Programmable GPU simulation framework |
| Taichi Lang / Taichi MPM | MLS-MPM, differentiable MPM, soft robotics, contact, cutting, and large deformation through applications and libraries | Particle/grid alternative to tetrahedral FEM; compact high-performance and differentiable implementations | Programmable language/runtime plus solver projects |
| Bullet `btSoftBody` | Cloth and volumetric soft bodies using engine-oriented particle/link/cluster representations | Practical interactive collision/contact baseline and legacy soft-body comparison | General physics engine with soft-body subsystem |
| NVIDIA PhysX 5 | GPU FEM deformable surfaces and volumes with separate simulation/collision meshes | Production FEM/contact architecture and GPU execution reference | General physics engine with FEM deformables |
| NVIDIA Isaac Sim | PhysX-based deformable objects exposed through a robotics simulation environment | Robotics, sensing, control, and interactive deployment context | Application platform built on PhysX |
| MuJoCo | Native `flex` deformable elements, including tetrahedral volumes, alongside rigid multibody dynamics | Control-oriented integration of rigid bodies, soft contact, and newer deformable elements | Robotics/control engine with deformable extension |
| SuperDex / Mochi | Tetrahedral soft volumes; experimental shells and rods; rigid/soft articulation | Contact-first implicit soft-body architecture, differentiable compliant contact, and solver composition | Contact-rich dexterous-manipulation engine |
| Drake | Tetrahedral FEM deformable bodies integrated with `MultibodyPlant` and deformable contact | Modular FEM, automatic differentiation, and rigid--deformable robotics integration | Robotics/control framework with FEM deformables |

## 12. Implications for SOFT

The current SOFT implementation is closest to the mechanics-first,
implicit-solver group:

- first-order tetrahedral FEM with lumped mass;
- Saint Venant--Kirchhoff and stable Neo-Hookean materials;
- semi-implicit Euler plus implicit backward Euler, midpoint, trapezoidal, and
  Newmark methods;
- matrix-free L-BFGS for the nonlinear position solve;
- endpoint element-Jacobian feasibility checks, but no collision/contact system.

The literature suggests three separate next steps rather than one generic
“make it interactive” change:

1. improve the implicit solver and timestep policy for stiff unconstrained
   dynamics;
2. add a collision/contact pipeline with an explicit choice of penalty,
   projection, complementarity, or augmented-Lagrangian response;
3. add a graphics-oriented bounded-budget mode whose acceptance criteria are
   finite state, bounded energy/displacement, and responsive latency rather
   than strict trajectory agreement.

This interpretation is consistent with the distinction between the scientific
and interactive autotuners. The scientific mode should continue to measure
trajectory and energy error, while the interactive mode can accept controlled
inexactness and numerical damping when the state remains bounded and usable.

## References and reading list

The links below are the primary sources used for the classifications above.

- [Terzopoulos, Platt, Barr, and Fleischer, “Elastically Deformable Models,” SIGGRAPH 1987](https://web.cs.ucla.edu/~dt/papers/siggraph87/siggraph87.pdf)
- [Terzopoulos and Fleischer, “Deformable Models,” The Visual Computer 1988](https://web.cs.ucla.edu/~dt/papers/viscomp88/viscomp88.pdf)
- [O'Brien and Hodgins, “Graphical Modeling and Animation of Brittle Fracture,” SIGGRAPH 1999](https://doi.org/10.1145/311535.311550)
- [Müller et al., “Stable Real-Time Deformations,” SCA 2002](https://doi.org/10.1145/545261.545263)
- [Irving, Teran, and Fedkiw, “Invertible Finite Elements for Robust Simulation of Large Deformation,” SCA 2004](https://doi.org/10.2312/SCA/SCA04/131-140)
- [Teran, Sifakis, Irving, and Fedkiw, “Robust Quasistatic Finite Elements and Flesh Simulation,” SCA 2005](https://graphics.cs.wisc.edu/Papers/2005/TSIF05/)
- [Sifakis, Shinar, Irving, and Fedkiw, “Hybrid Simulation of Deformable Solids,” SCA 2007](https://graphics.cs.wisc.edu/Papers/2007/SSIF07/)
- [Sifakis, Der, and Fedkiw, “Arbitrary Cutting of Deformable Tetrahedralized Objects,” SCA 2007](https://diglib.eg.org/server/api/core/bitstreams/cb0b56e2-3e56-4672-954a-f2aa92d510c9/content)
- [Sifakis and Barbic, “FEM Simulation of 3D Deformable Solids,” SIGGRAPH 2012 course](https://doi.org/10.1145/2343483.2343501)
- [Sifakis and Barbic, “FEM Simulation of 3D Deformable Solids,” course notes and companion material](https://viterbi-web.usc.edu/~jbarbic/femdefo/)
- [Rojas, Sifakis, and Kavan, “Differentiable Implicit Soft-Body Physics,” ICML 2021](https://arxiv.org/abs/2102.05791)
- [Baraff and Witkin, “Large Steps in Cloth Simulation,” SIGGRAPH 1998](https://doi.org/10.1145/280814.280821)
- [Bridson, Fedkiw, and Anderson, “Robust Treatment of Collisions, Contact and Friction for Cloth Animation,” SIGGRAPH 2002](https://doi.org/10.1145/566654.566623)
- [Selle, Su, Irving, and Fedkiw, “Robust High-Resolution Cloth Using Parallelism, History-Based Collisions, and Accurate Friction,” IEEE TVCG 2009](https://doi.org/10.1109/TVCG.2008.79)
- [Capell et al., “A Multiresolution Framework for Dynamic Deformations,” SCA 2002](https://grail.cs.washington.edu/projects/deformation/)
- [Teschner et al., “Collision Detection for Deformable Objects,” Eurographics STAR 2004](https://diglib.eg.org/items/120c9eda-8557-4563-b6df-5e3fa976a3e1)
- [Teschner et al., “A Versatile and Robust Model for Geometrically Complex Deformable Solids,” CGI 2004](https://cgweb.informatik.uni-freiburg.de/publications.htm)
- [Keiser et al., “Contact Handling for Deformable Point-Based Objects,” VMV 2004](https://matthias-research.github.io/pages/publications/cd_pba04.pdf)
- [Heidelberger et al., “Consistent Penetration Depth Estimation for Deformable Collision Response,” VMV 2004](https://cgweb.informatik.uni-freiburg.de/publications.htm)
- [Bielser, Glardon, and Teschner, “A State Machine for Real-Time Cutting of Tetrahedral Meshes,” Pacific Graphics 2003 / Graphical Models 2004](https://cgweb.informatik.uni-freiburg.de/publications.htm)
- [Teschner et al., “Optimized Spatial Hashing for Collision Detection of Deformable Objects,” VMV 2003](https://cgweb.informatik.uni-freiburg.de/publications.htm)
- [Faure et al., “SOFA, a Multi-Model Framework for Interactive Physical Simulation,” 2012](https://www.lirmm.fr/~gilles/papers/faure_springer12.pdf)
- [James and Pai, “ArtDefo: Accurate Real Time Deformable Objects,” SIGGRAPH 1999](https://graphics.stanford.edu/~djames/publication/artdefo-accurate-real-time-deformable-objects/)
- [James and Fatahalian, “Precomputing Interactive Dynamic Deformable Scenes,” SIGGRAPH 2003](https://graphics.stanford.edu/~djames/publication/precomputing-interactive-dynamic-deformable-scenes/)
- [James and Pai, “BD-Tree: Output-Sensitive Collision Detection for Reduced Deformable Models,” SIGGRAPH 2004](https://doi.org/10.1145/1015706.1015751)
- [Barbič and James, “Real-Time Subspace Integration for St. Venant-Kirchhoff Deformable Models,” SIGGRAPH 2005](https://graphics.cs.cmu.edu/projects/stvk/)
- [Kry, James, and Pai, “EigenSkin: Real-Time Large Deformation Character Skinning in Hardware,” SCA 2002](https://www.cs.mcgill.ca/~kry/pubs/eigenskin/)
- [Rivers and James, “FastLSM: Fast Lattice Shape Matching for Robust Real-Time Deformation,” SIGGRAPH/ACM TOG 2007](https://doi.org/10.1145/1276377.1276480)
- [Nesme, Kry, Jeřábková, and Faure, “Preserving Topology and Elasticity for Embedded Deformable Models,” SIGGRAPH/ACM TOG 2009](https://www.cs.mcgill.ca/~cg/projects/composite/)
- [Mercier-Aubin, Winter, Kry, and Levin, “Adaptive Rigidification of Elastic Solids,” SIGGRAPH/ACM TOG 2022](https://doi.org/10.1145/3528223.3530156)
- [Cai, Coevoet, Jacobson, and Kry, “Active Learning Neural C-space Signed Distance Fields for Reduced Deformable Self-Collision,” Graphics Interface 2022](https://www.cs.mcgill.ca/~kry/)
- [SOFA: an Open-source Solution for Physics Simulation](https://diglib.eg.org/server/api/core/bitstreams/11040985-c1bf-4547-84b6-2a436f9fede5/content)
- [SOFA Framework official website](https://www.sofa-framework.org/)
- [SOFA architecture and multi-model representation](https://www.sofa-framework.org/about-sofa/)
- [SOFA source repository](https://github.com/sofa-framework/sofa)
- [Müller et al., “Position Based Dynamics,” Eurographics/VRIPHYS 2006](https://doi.org/10.2312/PE/vriphys06/071-080)
- [Müller, Heidelberger, Teschner, and Gross, “Meshless Deformations Based on Shape Matching,” SIGGRAPH 2005](https://doi.org/10.1145/1073204.1073216)
- [Müller, “Hierarchical Position Based Dynamics,” VRIPHYS 2008](https://matthias-research.github.io/pages/publications/hpbd.pdf)
- [Müller and Chentanez, “Solid Simulation with Oriented Particles,” SIGGRAPH 2011](https://matthias-research.github.io/pages/publications/orientedParticles.pdf)
- [Müller, Chentanez, Kim, and Macklin, “Strain Based Dynamics,” SCA 2014](https://matthias-research.github.io/pages/publications/strainBasedDynamics.pdf)
- [Müller, Chentanez, Kim, and Macklin, “Air Meshes for Robust Collision Handling,” SIGGRAPH 2015](https://matthias-research.github.io/pages/publications/airMeshesPreprint.pdf)
- [Galoppo et al., “Fast Simulation of Deformable Models in Contact Using Dynamic Deformation Textures,” SCA 2006](https://diglib.eg.org/bitstreams/d6f0958c-f0f4-4f85-baaa-613176c84268/download)
- [Barbič and James, “Time-Critical Distributed Contact for 6-DoF Haptic Rendering of Adaptively Sampled Reduced Deformable Models,” SCA 2007](https://diglib.eg.org/bitstreams/a4e24227-0851-4a9b-ab3c-d55282124551/download)
- [Shinar, Schroeder, and Fedkiw, “Two-way Coupling of Rigid and Deformable Bodies,” SCA 2008](https://diglib.eg.org/items/54af7e32-56a8-42f0-be42-4b5363f5f232)
- [Otaduy et al., “Implicit Contact Handling for Deformable Objects,” Eurographics/CGF 2009](https://doi.org/10.1111/j.1467-8659.2009.01396.x)
- [Miguel and Otaduy, “Efficient Simulation of Contact between Rigid and Deformable Objects,” 2011](https://www.researchgate.net/publication/228448338_EFFICIENT_SIMULATION_OF_CONTACT_BETWEEN_RIGID_AND_DEFORMABLE_OBJECTS)
- [Garre and Otaduy, “Haptic Rendering of Objects with Rigid and Deformable Parts,” Computers & Graphics 2010](https://www.sciencedirect.com/science/article/abs/pii/S0097849310001350)
- [Miguel et al., “Modeling and Estimation of Energy-Based Hyperelastic Objects,” Computer Graphics Forum 2016](https://onlinelibrary.wiley.com/doi/10.1111/cgf.12840)
- [Kim et al., “FEM Simulation of 3D Deformable Solids,” SIGGRAPH 2012 Course](https://doi.org/10.1145/2343483.2343501)
- [Bargteil and Cohen, “Animation of Deformable Bodies with Quadratic Bézier Finite Elements,” ACM TOG 2014](https://doi.org/10.1145/2567943)
- [Bender, Koschier, Charrier, and Weber, “Position-Based Simulation of Continuous Materials,” Computers & Graphics 2014](https://doi.org/10.1016/j.cag.2014.07.004)
- [Bouaziz et al., “Projective Dynamics,” SIGGRAPH/ACM TOG 2014](https://doi.org/10.1145/2601097.2601116)
- [Li et al., “Deformable Objects Collision Handling with Fast Convergence,” Eurographics/CGF 2015](https://doi.org/10.1111/cgf.12765)
- [Macklin, Müller, and Chentanez, “XPBD,” MIG 2016](https://doi.org/10.1145/2994258.2994272)
- [Müller, Bender, Chentanez, and Macklin, “A Robust Method to Extract the Rotational Part of Deformations,” Motion in Games 2016](https://matthias-research.github.io/pages/publications/rotation.pdf)
- [Chentanez, Müller, Macklin, and Kim, “Real-Time Simulation of Large Elasto-Plastic Deformation with Shape Matching,” SCA 2016](https://matthias-research.github.io/pages/publications/elastoplastic.pdf)
- [Macklin et al., “Non-Smooth Newton Methods for Deformable Multi-Body Dynamics,” SIGGRAPH/ACM TOG 2019](https://doi.org/10.1145/3338695)
- [Macklin et al., “Small Steps in Physics Simulation,” SCA 2019](https://doi.org/10.1145/3309486.3340247)
- [Macklin and Müller, “A Constraint-Based Formulation of Stable Neo-Hookean Materials,” MIG 2021](https://matthias-research.github.io/pages/publications/neohookean.pdf)
- [Barbič and James, “Six-DoF Haptic Rendering of Contact between Geometrically Complex Reduced Deformable Models,” IEEE Transactions on Haptics 2008](https://graphics.stanford.edu/~djames/research/)
- [Kim and James, “Skipping Steps in Deformable Simulation with Online Model Reduction,” SIGGRAPH Asia 2009](https://doi.org/10.1145/1618452.1618469)
- [Smith, de Goes, and Kim, “Stable Neo-Hookean Flesh Simulation,” ACM TOG 2018](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf)
- [Kim, de Goes, and Iben, “Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation,” SIGGRAPH/ACM TOG 2019](https://www.tkim.graphics/ANISO/)
- [Kim and Eberle, “Dynamic Deformables: Implementation and Production Practicalities,” SIGGRAPH Courses 2020/2022](https://doi.org/10.1145/3532720.3535628)
- [HOBAK: A Library for Squashing Things](https://github.com/theodorekim/HOBAKv1)
- [Lin, Chitalu, and Komura, “Isotropic ARAP Energy Using Cauchy-Green Invariants,” ACM TOG 2022](https://doi.org/10.1145/3550454.3555507)
- [Chitalu, Dubach, and Komura, “Binary Ostensibly-Implicit Trees for Fast Collision Detection,” Eurographics/CGF 2020](https://www.pure.ed.ac.uk/ws/files/142704991/Binary_Ostensibly_Implicit_CHITALU_DOA17022020_AFV.pdf)
- [Chitalu, Dubach, and Komura, “Bulk-Synchronous Parallel Simultaneous BVH Traversal for Collision Detection on GPUs,” I3D 2018](https://doi.org/10.1145/3190834.3190848)
- [Chitalu, Miao, Subr, and Komura, “Displacement-Correlated XFEM for Simulating Brittle Fracture,” Computer Graphics Forum 2020](https://doi.org/10.1111/cgf.13953)
- [Fan, Chitalu, and Komura, “Simulating Brittle Fracture with Material Points,” ACM TOG 2022](https://doi.org/10.1145/3522573)
- [Taku Komura publication list](https://i.cs.hku.hk/~taku/publication.html)
- [Schneider et al., “Poly-Spline Finite-Element Method,” ACM TOG 2019](https://doi.org/10.1145/3313797)
- [Longva et al., “Pitfalls of Projection: A Study of Newton-Type Solvers for Incremental Potentials,” 2023](https://arxiv.org/abs/2311.14526)
- [Li et al., “Incremental Potential Contact,” SIGGRAPH/ACM TOG 2020](https://ipc-sim.github.io/)
- [Ton-That, Kry, and Andrews, “Generalized eXtended Finite Element Method for Deformable Cutting via Boolean Operations,” Computer Graphics Forum 2024](https://doi.org/10.1111/cgf.15184)
- [PolyFEM documentation and source](https://polyfem.github.io/)
- [Bender, Müller, Otaduy, Teschner, and Macklin, “A Survey on Position-Based Simulation Methods in Computer Graphics”](https://animation.rwth-aachen.de/publication/0512/)
- [Chen et al., “Vertex Block Descent,” SIGGRAPH/ACM TOG 2024](https://doi.org/10.1145/3658179)
- [Giles, Diaz, and Yuksel, “Augmented Vertex Block Descent,” SIGGRAPH/ACM TOG 2025](https://doi.org/10.1145/3731195)
- [Fernández-Fernández, Löschner, and Bender, “Progressively Projected Newton's Method,” Computer Graphics Forum 2026](https://doi.org/10.1111/cgf.70386)
- [Müller, Macklin, Chentanez, and Jeschke, “Physically Based Shape Matching,” Computer Graphics Forum 2022](https://doi.org/10.1111/cgf.14618)
- [Hu et al., “A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling,” SIGGRAPH/ACM TOG 2018](https://yuanming.taichi.graphics/publication/2018-mlsmpm/)
- [Hu et al., “ChainQueen: A Real-Time Differentiable Physical Simulator for Soft Robotics,” ICRA 2019](https://yuanming.taichi.graphics/publication/2019-chainqueen/)
- [Hu, “Taichi: A Language for High-Performance Computation on Spatially Sparse Data Structures,” SIGGRAPH Asia 2019](https://yuanming.taichi.graphics/publication/2019-taichi/)
- [Hu et al., “DiffTaichi: Differentiable Programming for Physical Simulation,” ICLR 2020](https://arxiv.org/abs/1910.00935)
- [Macklin, “Warp: A High-Performance Python Framework for GPU Simulation and Graphics,” NVIDIA GTC 2022](https://github.com/NVIDIA/warp)
- [NVIDIA Warp documentation and FEM toolkit](https://nvidia.github.io/warp/stable/index.html)
- [Bullet Physics source repository](https://github.com/bulletphysics/bullet3)
- [Bullet user manual: soft-body dynamics](https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/Bullet_User_Manual.pdf)
- [NVIDIA PhysX deformable-body documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/DeformableBodyOverview.html)
- [NVIDIA Isaac Lab deformable-object tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_deformable_object.html)
- [Todorov, Erez, and Tassa, “MuJoCo: A Physics Engine for Model-Based Control,” IROS 2012](https://doi.org/10.1109/IROS.2012.6386109)
- [MuJoCo modeling documentation: deformable objects and `flex`](https://mujoco.readthedocs.io/en/3.3.3/modeling.html)
- [Project SuperDex](https://projectsuperdex.com/)
- [SuperDex actors overview](https://projectsuperdex.com/physics/docs/concepts/actors/overview/)
- [SuperDex materials](https://projectsuperdex.com/physics/docs/category/materials/)
- [SuperDex contact model](https://projectsuperdex.com/physics/docs/concepts/contact/)
- [SuperDex solvers](https://projectsuperdex.com/physics/docs/concepts/solvers/)
- [Project SuperDex source repository](https://github.com/facebookresearch/project_superdex)
- [Drake deformable-body API](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_deformable_body.html)
- [Drake FEM model](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1fem_1_1_fem_model.html)
- [Drake FEM configuration](https://drake.mit.edu/pydrake/pydrake.multibody.fem.html)
- [Todorov et al., “A Convex Formulation of Frictional Contact between Rigid and Deformable Bodies”](https://arxiv.org/abs/2303.08912)
- [Drake source repository](https://github.com/RobotLocomotion/drake)
