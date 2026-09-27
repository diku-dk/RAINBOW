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

## What `soft` currently implements

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

The implementation map is maintained in the [soft architecture](architecture.md),
[numerics](numerics.md), [time-stepping](time-stepping.md), and
[extensions](extensions.md) pages. This review supplies the external evidence
used to assess those choices.

## Executive findings

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

## Cross-paper comparison

The table summarizes representative works. “Not reported” means that the
paper or accessible implementation does not provide enough information for a
reliable classification; it should not be interpreted as “not used.”

| Work | Venue/year | Representation and material | Time stepping / solve | Collision detection | Contact resolution | Interactive regime |
|---|---|---|---|---|---|---|
| Terzopoulos, Platt, Barr, Fleischer, *Elastically Deformable Models* | SIGGRAPH 1987 | Deformable curves, surfaces, and solids; elasticity-based models | Numerical solution of coupled dynamic equations; implementation details reflect early continuum/mechanical simulation | Impenetrable obstacles and applied constraints | Force/constraint treatment in the deformable-model formulation | Primarily animation/offline; foundational rather than a modern real-time system |
| Terzopoulos and Fleischer, *Deformable Models* | The Visual Computer 1988 | Curves, surfaces, and solids; elastic and inelastic extensions | Dynamic differential equations; implicit methods are part of the early deformable-model lineage | Obstacles and external constraints | Forces and constraints; inelastic reference-state evolution for viscoelasticity/plasticity/fracture | Animation and modeling; interactive manipulation is a goal, but no modern frame-budget claim |
| O'Brien and Hodgins, *Graphical Modeling and Animation of Brittle Fracture* | SIGGRAPH 1999 | Volumetric finite elements with fracture-dependent topology | Explicit dynamic update in the fracture animation pipeline | Geometric fracture and newly exposed surfaces | Element separation and topology changes rather than persistent contact mechanics | Offline animation; important because topology change is a separate problem from ordinary deformation |
| Müller et al., *Stable Real-Time Deformations* lineage | SIGGRAPH/ACM TOG 2002--2004 | Volumetric tetrahedral FEM, commonly linear tetrahedra with corotational or modal acceleration | Explicit or reduced/modal stepping in interactive examples; solver details vary by paper | Collision proxies and application-specific geometric tests | Penalty, constraints, or application-specific response; no single unified contact method | Interactive reduced-resolution regime; the key gain is reducing the dynamic state, not making full nonlinear FEM cheap |
| Irving, Teran, and Fedkiw, *Invertible Finite Elements for Robust Simulation of Large Deformation* | SCA 2004 | Linear tetrahedral FEM with an invertibility-preserving elastic formulation | Implicit dynamics and robust force evaluation around degenerate or inverted configurations | Geometry-dependent; element robustness is the central contribution | Stabilized elastic forces and collision handling in the surrounding simulator | Robustness-oriented mechanics; a building block for larger-step solvers, not itself a contact architecture |
| Baraff and Witkin, *Large Steps in Cloth Simulation* | SIGGRAPH 1998 | Triangular cloth mesh; continuum-inspired stretch, shear, bending, and damping | Implicit Euler; modified constrained conjugate gradient solve; adaptive step-size logic | Coherency-based bounding boxes and cloth collision tests | Direct constraint enforcement/position alteration, with constraints maintained during the iterative solve | Interactive-oriented CPU cloth; large steps are the central contribution, but scale and hardware are historical |
| Bridson, Fedkiw, Anderson, *Robust Treatment of Collisions, Contact and Friction for Cloth Animation* | SIGGRAPH/ACM TOG 2002 | Particle/triangle cloth; internal dynamics are deliberately decoupled from collision handling | Compatible with the underlying cloth integrator; collision method is solver-independent | Geometric collision tests with thickness and continuous/robust handling for remaining collisions | Non-stiff repulsion forces plus fail-safe collision processing; static and kinetic friction | Production-animation quality and robust cloth; collision cost, not only integration, limits interactive scale |
| Selle et al., *Robust High-Resolution Cloth Using Parallelism, History-Based Collisions, and Accurate Friction* | IEEE TVCG 2009 | Triangular cloth mesh; edge stretch springs, bending springs, and damping | Modified time integration with distributed parallel evolution; the paper discusses semi-implicit and fully implicit choices rather than reducing the result to one universal integrator label | AABB hierarchy plus history-based proximity tests and collision state | History-based repulsion/attraction, Gauss--Seidel collision response, accurate friction, and rigid impact zones for unresolved collisions | High-resolution/hero-cloth production regime; scales to very large meshes, but is not an ordinary frame-rate interactive method |
| Capell et al., *A Multiresolution Framework for Dynamic Deformations* | SCA 2002 | Volumetric subdivision hierarchy and embedded domains for elastic solids | Implicit solver for large stable timesteps | Not the primary contribution; geometric embedding and constraints support interaction | Constraints on the reduced/hierarchical deformation representation | Interactive-oriented reduced deformation; high-resolution detail is introduced selectively |
| Teschner et al., *Collision Detection for Deformable Objects* | Eurographics STAR 2004 | Survey covering meshes, particles, and deformable surfaces | Not a single integrator | Bounding-volume hierarchies, distance fields, spatial partitioning, and related broad/narrow phases | Survey of penalty, constraints, and application-specific response | Focuses on interactive collision workloads in surgery and entertainment |
| Müller et al., *Position Based Dynamics* | VRIPHYS/Eurographics 2006 | Particles and geometric constraints; material behavior encoded through constraints | Position update with iterative constraint projection; velocity layer can be omitted | Collision constraints generated from geometric proximity/intersection | Sequential or parallel projection to valid positions; penetration is removed directly | Strongly interactive: bounded iterations provide predictable cost and controllability |
| Galoppo et al., *Fast Simulation of Deformable Models in Contact Using Dynamic Deformation Textures* | SCA 2006 | Layered deformable bodies with a rigid/articulated core and surface deformation textures | Implicit, parallelizable formulation with large steps | Two-stage, output-sensitive proximity queries on dynamic deformation textures | Constraint forces using Lagrange multipliers and approximate implicit integration | Designed for high-resolution contact and real-time shading; SIMD/commodity-hardware oriented |
| Barbič and James, *Time-Critical Distributed Contact for 6-DoF Haptic Rendering of Adaptively Sampled Reduced Deformable Models* | SCA 2007 | Reduced deformable models with adaptively sampled surface points | Fast reduced-state time stepping followed by contact evaluation | Sphere-tree / pointshell hierarchy and adaptive sampling | Time-critical contact force estimation for haptics | Haptic regime: contact must meet a much tighter loop than ordinary graphics frame rate |
| Otaduy et al., *Implicit Contact Handling for Deformable Objects* | Eurographics/CGF 2009 | Deformable solids and cloth; heterogeneous deformables | Implicit constrained dynamics with large timesteps | Collision/contact detection feeding inequality constraints | Implicit complementarity constraints solved with iterative constraint anticipation and an MLCP solver | Interactive-oriented robust contact, including stacking, rolling, sliding, and high-speed examples |
| Shinar, Schroeder, Fedkiw, *Two-way Coupling of Rigid and Deformable Bodies* | SCA 2008 | Rigid bodies plus deformable bodies, including thin shells and arbitrary constitutive models | Unified time-integration and two-way coupled algorithms | Rigid/deformable and self-collision handling | Coupled contact, stacking, friction, and constraints | Interactive-capable coupled simulation; complexity grows with contact and model resolution |
| Bargteil and Cohen, *Animation of Deformable Bodies with Quadratic Bézier Finite Elements* | ACM TOG 2014 | Quadratic Bézier tetrahedral elements, optionally mixed adaptively with linear elements; corotational linear strain | Dynamic FEM with adaptive degree and efficient precomputation; exact timestep details are implementation-dependent | Standard geometric collision mechanisms; not the contribution | Application-dependent contact response | Interactive-oriented higher-order FEM; improves rest-shape/detail efficiency but increases element and integration complexity |
| Bender et al., *Position-Based Simulation of Continuous Materials* | Computers & Graphics 2014 | Continuum-based strain and bending energies on cloth and volumetric bodies; anisotropy and elastoplasticity | Iterative position-based energy reduction rather than force-based Newton dynamics | Geometric proximity and constraint tests | Position projection with material constraints; examples include degenerate/inverted elements | Explicitly interactive: thousands of degrees of freedom with controllable, bounded iterations |
| Bouaziz et al., *Projective Dynamics* | SIGGRAPH/ACM TOG 2014 | FEM-inspired constraints for solids, cloth, shells, and examples | Implicit Euler reformulated as alternating local projections and a prefactorized global solve | Application-dependent collision constraints; core method focuses on elastic constraints | Projection-based constraints in an optimization framework | Strong interactive regime: reports 49k DoFs at about 3.1 ms per iteration and 10 iterations per frame in a representative example |
| Macklin, Müller, Chentanez, *XPBD* | MIG 2016 | Particles and compliant constraints representing elastic and dissipative potentials | Position-based iterative solve with compliance; time-step/iteration dependence is reduced | Collision constraints in the same position-constraint framework | Compliant constraint projection with persistent constraint behavior | Interactive real-time regime; quality and stiffness are traded against iteration budget |
| Smith, de Goes, and Kim, *Stable Neo-Hookean Flesh Simulation* | SIGGRAPH/ACM TOG 2018 | Hexahedral lattice and hyperelastic Neo-Hookean family; near-incompressible flesh with Poisson ratio near 0.5 | Quasistatic/dynamic implicit Newton with CG; element Hessians are projected to positive semidefinite form | Collision is application-dependent; the contribution is constitutive and solver robustness | Solver-compatible constraints/contact in the surrounding character simulator | High-fidelity production/offline character simulation; one example has 45,809 elements and 156,078 DoFs, with 25.6 s per average timestep |
| Li et al., *Deformable Objects Collision Handling with Fast Convergence* | Eurographics/CGF 2015 | Deformable objects with large deformation; elastic energy formulation | Optimization derived from implicit integration | Collision inequalities, with redundant constraints pruned | General linear constraints solved with an MPRGP-derived method | Interactive/stable large-deformation target; reports an order-of-magnitude speedup over ICA in its experiments |
| Li et al., *Incremental Potential Contact* | SIGGRAPH/ACM TOG 2020 | FEM and other discretizations; contact-safe incremental potential formulation | Implicit Euler expressed as an incremental potential with barrier terms and line search | Continuous collision detection through barrier activation and distance queries | Barrier-based non-penetration and friction without penalty stiffness | Robust large-step contact; moderate models may be interactive, but contact and nonlinear iterations remain dominant |
| Schneider et al., *Poly-Spline Finite-Element Method* and PolyFEM | ACM TOG 2019; open-source library | High-order triangles/tetrahedra, splines, polygons/polyhedra; PolyFEM supports multiple constitutive models and orders beyond linear | Library-level transient integrators and nonlinear solves; configuration-dependent rather than one prescribed method | Problem-dependent collision and boundary handling | Problem-dependent constraints/contact; a reference architecture for moving beyond linear-tet-only assumptions | Scientific/engineering and research-prototyping regime; flexibility and accuracy are prioritized over one fixed real-time path |
| Ton-That, Kry, and Andrews, *Generalized eXtended Finite Element Method for Deformable Cutting via Boolean Operations* | Computer Graphics Forum 2024 | XFEM enrichment decouples cut geometry from the background simulation mesh | FEM integration with discontinuous integration over cut subdomains | Cut geometry represented by robust mesh Booleans, not only collision pairs | Discontinuous enriched basis and local integration preserve deformation across cuts | Interactive/surgical-extension regime; local cut cost avoids global remeshing but adds enrichment and integration complexity |
| Fernández-Fernández, Löschner, and Bender, *Progressively Projected Newton's Method* | Computer Graphics Forum 2026 | Element-based hyperelastic deformables, including contact-rich examples | Newton with selective Hessian projection; compared with Projected Newton and Project-on-Demand Newton over timesteps, tolerances, and resolutions | Contact-rich tests use the surrounding deformable-contact pipeline | Projected/conditional Hessian strategies provide descent directions for nonlinear solves | State-of-the-art solver regime; emphasizes fewer eigendecompositions and faster convergence, not a new material or contact model |
| Chen et al., *Vertex Block Descent* | SIGGRAPH/ACM TOG 2024 | Volumetric elastic bodies, including linear tetrahedral FEM | Variational implicit Euler solved by vertex-level Gauss--Seidel/block descent; mesh coloring and GPU parallelism | Parallel collision processing integrated with the solver | Constraint, collision, and friction formulations; bounded iterations retain stability | Modern GPU interactive-to-large-scale regime; some reported scenes are seconds per step, while smaller scenes target real-time use |
| Giles, Diaz, Yuksel, *Augmented Vertex Block Descent* | SIGGRAPH/ACM TOG 2025 | Elastic bodies and coupled rigid/deformable systems | VBD plus augmented-Lagrangian updates; implicit-Euler target | GPU collision processing | Hard constraints, stacking, friction, and attachments through augmented Lagrangians | GPU real-time/stable regime, including very large contact scenes; performance depends strongly on hardware and iteration budget |

## Assessment of the `soft` design choices

### Spatial discretization: linear tetrahedra are representative, but not sufficient

Linear tetrahedra are a defensible first implementation for `soft`. They are
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

| `soft` spatial choice | Evidence in reviewed work | Assessment and consequence |
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

| Material family | Where it is representative | Typical evidence or settings | Relevance to `soft` |
|---|---|---|---|
| StVK / Green-strain elasticity | Baseline nonlinear FEM, moderate strain, educational and reference implementations | Often selected for simple parameterization; exact physical parameters are frequently scene-specific or omitted | Keep for verification, patch tests, and controlled nonlinear studies; label it as a model limitation in large-strain examples |
| Corotational linear elasticity | Real-time volumetric graphics and interactive FEM | Uses linear strain in a rotating local frame; efficient constant element stiffness is a major attraction | Useful comparison model, but current `soft` should not infer that corotational and finite-strain hyperelasticity are interchangeable |
| Stable Neo-Hookean | Flesh, high rotation, large deformation, near-incompressible tissue | Smith et al. use ν = 0.488 in a representative flesh comparison and analyze positive-semidefinite Hessian projections | Strong current choice; add parameterized tests over ν, shear modulus/Young's modulus, and inversion states |
| Mooney--Rivlin, Arruda--Boyce, Fung, anisotropic models | Biological tissue, rubber-like materials, anisotropic tissue, advanced production | Kim et al. compare several hyperelastic families; Bender et al. support anisotropy and elastoplasticity | Future constitutive plug-in interface; validate stress, tangent, and conditioning independently of the stepper |
| XPBD compliance / constraint materials | Interactive graphics where stiffness must be controllable under bounded iterations | Compliance absorbs timestep/iteration dependence rather than representing a unique continuum modulus | Relevant only if `soft` adds a constraint-oriented interactive branch; do not silently compare compliance to Young's modulus |

The literature does not support a single “typical” numerical material value
for generic soft bodies. Density, Young's or shear modulus, Poisson ratio,
damping, thickness, and friction are normally chosen from the application:
cloth, flesh, rubber, surgical tissue, and animation props have different
scales. A useful `soft` benchmark should therefore publish dimensionless
ratios or nondimensional conditioning indicators alongside SI values, for
example stiffness-to-inertia ratio, ν, timestep relative to the fastest mode,
and contact thickness relative to element size.

### Time stepping and nonlinear solvers: the current coverage is good but misses the dominant contact solvers

The current `soft` time-stepping coverage is unusually good for a small
reference implementation: semi-implicit Euler, backward Euler, implicit
midpoint, trapezoidal, and average-acceleration Newmark cover first-order
stable dynamics, second-order low-dissipation dynamics, and a structural-
dynamics formulation. The same position/residual machinery feeds the
matrix-free L-BFGS nonlinear solve.

The field coverage is not complete. The most important omissions are not
another named one-step formula, but alternative nonlinear and contact solve
families:

| Family | Representative works | What it adds beyond current `soft` |
|---|---|---|
| Newton / Projected Newton | Teran et al.; Smith--de Goes--Kim; Longva et al.; Fernández-Fernández et al. | Exact or projected Hessian information, quadratic local convergence, and explicit handling of indefinite element Hessians |
| Project-on-Demand / Kinetic Newton | Longva et al., *Pitfalls of Projection* | Robustness only where needed, avoiding the cost and convergence damage of projecting every element Hessian |
| Progressively Projected Newton | Fernández-Fernández, Löschner, Bender | A current state-of-the-art projected-Newton variant; uses residual information to project a small subset of element Hessians |
| Projective Dynamics | Bouaziz et al. | Local/global split and prefactorized global matrix; attractive when energies have the required projective form |
| Vertex Block Descent / AVBD | Chen et al.; Giles et al. | Local block updates, mesh coloring, GPU execution, and augmented-Lagrangian hard constraints |
| Complementarity / nonsmooth Newton | Otaduy et al.; Macklin et al.; Deul et al. | Contact-aware solves for non-penetration, friction, and changing active sets |
| Barrier-based incremental potentials | Li et al., *Incremental Potential Contact* | Continuous collision-safe contact without tuning a penalty stiffness; couples CCD, barrier energy, friction, and line search |
| Higher-order FEM and p-multigrid | Bargteil--Cohen; PolyFEM; cubic-element/p-multigrid work | More accuracy per element and a route to curved geometry, at the cost of richer quadrature and preconditioning |

Thus, `soft` has broad *time-integrator* coverage but narrow *nonlinear
solver/contact* coverage. Adding BDF2 or generalized-α would be valuable for
specific numerical-damping and multistep studies, but it is less urgent for
field coverage than adding a sparse Newton/projected-Newton comparison and a
contact-safe incremental-potential path.

### Explicit links to the current implementation

| Current code/documentation | Literature concept it corresponds to | What should be compared next |
|---|---|---|
| `rainbow/simulators/prox_soft/time_stepper.py` | One-step and structural time integrators | Refinement studies for all five current methods, then BDF2/generalized-α only if a use case requires them |
| `rainbow/simulators/prox_soft/nonlinear.py` | Matrix-free L-BFGS, Armijo globalization, fallback/watchdog/rescue | Projected Newton, Project-on-Demand Newton, and barrier/contact line searches |
| `rainbow/simulators/prox_soft/solver.py` | SoftBody state, force/residual assembly, NumPy/JAX execution | Separate mechanical residual, contact residual, collision broad phase, and backend-independent diagnostics |
| `rainbow/simulators/prox_soft/material.py` | StVK and stable Neo-Hookean constitutive kernels | Constitutive plug-ins with stress/tangent/energy verification and parameterized near-incompressible tests |
| `rainbow/geometry/volume_mesh.py` | Structured tetrahedral generation and surface extraction | Mesh quality metrics, curved/high-order elements, remeshing, and cut-aware topology/enrichment |
| No current module | Collision, contact, friction, cutting | Start with surface extraction plus broad/narrow phase; select barrier, complementarity, or XPBD-style response as separate policies |

## Collision/contact design assessment for a future `soft` extension

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

The recommended first extension for a mechanics-first `soft` is a surface
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
extension point for `soft` than embedding cut-specific topology edits inside
the current tetrahedral force kernel.

## Groups of approaches

### 1. Early continuum and physically based deformable models

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

### 2. Implicit large-step mechanics

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

### 3. Robust cloth collision and contact

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

### 4. Constraint and position-based simulation

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

### 5. Reduced and hierarchical representations

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

### 6. Optimization and local-update implicit solvers

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

## Collision detection and contact resolution patterns

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

## Materials and parameter practice

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

For the current RAINBOW soft-body prototype, this suggests separating three
claims: constitutive correctness, trajectory accuracy, and interactive
robustness. A method can satisfy the third through numerical damping or
bounded inexact solves while failing the first two for a scientific undamped
test.

## What “interactive” means in the reviewed field

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

## Open-source systems and practical references

### SOFA

[SOFA](https://github.com/sofa-framework/sofa) is an open-source framework
targeted at interactive multiphysics, especially medical simulation and
robotics. Its [collision documentation](https://sofa-framework.github.io/doc/simulation-principles/multi-model-representation/collision/)
describes a modular pipeline with broad phase, narrow phase, contact output,
and selectable penalty, persistent, or constraint-based responses. Its
[simulation tutorial](https://sofa-framework.github.io/doc/simulation-principles/example-simple-body/)
shows the separation between integration scheme, linear solver, mechanical
model, and collision response. SOFA is valuable here because it demonstrates
how a production research framework exposes solver and contact choices as
components rather than baking them into one deformable-body algorithm.

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

### Tutorials and implementation material

The [Baraff and Witkin paper](https://www.cs.cmu.edu/~baraff/papers/sig98.pdf),
the [PBD paper](https://doi.org/10.2312/PE/vriphys06/071-080), the
[XPBD paper](https://matthias-research.github.io/pages/publications/XPBD.pdf),
and the [Projective Dynamics paper](https://www.projectivedynamics.org/projectivedynamics.pdf)
are unusually useful implementation references because they expose the
algorithmic structure rather than only reporting application results.

## Implications for the current soft-body prototype

The current RAINBOW implementation is closest to the mechanics-first,
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
- [Baraff and Witkin, “Large Steps in Cloth Simulation,” SIGGRAPH 1998](https://doi.org/10.1145/280814.280821)
- [Bridson, Fedkiw, and Anderson, “Robust Treatment of Collisions, Contact and Friction for Cloth Animation,” SIGGRAPH 2002](https://doi.org/10.1145/566654.566623)
- [Selle, Su, Irving, and Fedkiw, “Robust High-Resolution Cloth Using Parallelism, History-Based Collisions, and Accurate Friction,” IEEE TVCG 2009](https://doi.org/10.1109/TVCG.2008.79)
- [Capell et al., “A Multiresolution Framework for Dynamic Deformations,” SCA 2002](https://grail.cs.washington.edu/projects/deformation/)
- [Teschner et al., “Collision Detection for Deformable Objects,” Eurographics STAR 2004](https://diglib.eg.org/items/120c9eda-8557-4563-b6df-5e3fa976a3e1)
- [Müller et al., “Position Based Dynamics,” Eurographics/VRIPHYS 2006](https://doi.org/10.2312/PE/vriphys06/071-080)
- [Galoppo et al., “Fast Simulation of Deformable Models in Contact Using Dynamic Deformation Textures,” SCA 2006](https://diglib.eg.org/bitstreams/d6f0958c-f0f4-4f85-baaa-613176c84268/download)
- [Barbič and James, “Time-Critical Distributed Contact for 6-DoF Haptic Rendering of Adaptively Sampled Reduced Deformable Models,” SCA 2007](https://diglib.eg.org/bitstreams/a4e24227-0851-4a9b-ab3c-d55282124551/download)
- [Shinar, Schroeder, and Fedkiw, “Two-way Coupling of Rigid and Deformable Bodies,” SCA 2008](https://diglib.eg.org/items/54af7e32-56a8-42f0-be42-4b5363f5f232)
- [Otaduy et al., “Implicit Contact Handling for Deformable Objects,” Eurographics/CGF 2009](https://doi.org/10.1111/j.1467-8659.2009.01396.x)
- [Bouaziz et al., “Projective Dynamics,” SIGGRAPH/ACM TOG 2014](https://doi.org/10.1145/2601097.2601116)
- [Li et al., “Deformable Objects Collision Handling with Fast Convergence,” Eurographics/CGF 2015](https://doi.org/10.1111/cgf.12765)
- [Macklin, Müller, and Chentanez, “XPBD,” MIG 2016](https://doi.org/10.1145/2994258.2994272)
- [Chen et al., “Vertex Block Descent,” SIGGRAPH/ACM TOG 2024](https://doi.org/10.1145/3658179)
- [Giles, Diaz, and Yuksel, “Augmented Vertex Block Descent,” SIGGRAPH/ACM TOG 2025](https://doi.org/10.1145/3731195)
