# Changelog 

## Unreleased

### Added
- `Assembly` couples several models through kinematic constraints, mechanical or thermal, in two or three dimensions: `coupling(...)` makes selected nodes follow the rigid-body motion of the nearest node of another part, optionally on selected degrees of freedom.
- `ReferencePoint` and `ReferencePointHeat` enter an assembly as free nodes carrying rigid-body degrees of freedom or one temperature, so a load or a prescribed value applied there drives everything coupled to it.
- `Assembly.plot(...)` draws every part into one figure, with the points as markers and each coupling between the nodes it pairs, dispatching to matplotlib in 2D and PyVista in 3D. An argument given as a list is spread over the parts, so the `u` of `solve(...)` passes straight through.
- `mesh_to_lattice(...)` turns a planar or solid mesh into a lattice of bar elements along its edges, optionally bracing each quadrilateral with one diagonal (`"up"`, `"down"`) or both (`"cross"`).
- `Element.edges` lists the local node indices of the element edges, carrying the mid-side node on quadratic elements, and `linear_etype(...)` infers the linear element type of a mesh.

### Changed
- `Material.vectorize(...)` returns the type it was called on rather than a `Material`, so the properties of a concrete material stay visible to a type checker after vectorizing.
- `PlanarHeat` and `SolidHeat` no longer inherit from `Planar` and `Solid`, but share their elements, integration and plotting through a `PlanarGeometry` and a `SolidGeometry` base that carries no physics. A thermal model therefore no longer exposes `forces`, `displacements`, `ext_strain` and `solve_modes`, which belonged to mechanics and aliased the thermal `heat_flux` and `temperatures`. An external gradient stays zero for a thermal model.
- `k0(...)` is implemented once on `FEM` instead of separately for mechanics and heat, which built the same tensors under different names, and returns the same values as before.
- `Laminate.plot(...)` takes an `ax` to draw into, as the other matplotlib plotters do, and leaves showing the figure to the caller.
- `Laminate(...)` takes its layers as sequences rather than lists, so a tuple, or a list of one concrete material class held in a variable, is accepted where the invariance of `list` rejected it before.
- `Truss.plot3d(...)` draws all bars in one pass instead of one mesh each, so a lattice of a few thousand bars renders in under a second instead of minutes. The spheres at the joints follow the largest bar meeting there instead of the mean of all bars.
- `Planar.plot(...)` fills its elements, so a mesh reads as a solid rather than a lattice. `color` sets that fill and defaults to light blue as in the PyVista plotters, edges and labels follow the foreground of the matplotlib style, and `**kwargs` now reach the element `PolyCollection` instead of being ignored, so `edgecolor` or `hatch` work and a misspelled argument raises.
- Boundary conditions in the matplotlib plotters are outlined, dotted where they attach to a node, and sized in points rather than in a fraction of the model. They sit above the node markers, with the arrows above the constraint markers they may cross.
- `Material.rotate(...)` returns a rotated material instead of rotating the material in place, so a material may be rotated repeatedly, or shared between several rotated copies, without the rotations accumulating. Code that called it for its effect, as in `material.rotate(R)`, has to take the result now: `material = material.rotate(R)`.
- `Material.vectorize(...)` is implemented once on the base class, batching the properties a material holds instead of rebuilding it through its constructor, rather than being written out by every material itself. It no longer prints when it is called on a material that is already vectorized, and it keeps the type of a `TransverseIsotropicElasticity3D` instead of returning an `OrthotropicElasticity3D`.

### Fixed
- The two `Negative Jacobian` checks of `Truss` and `Shell`, the unsupported element type of `linear_etype(...)`, and the two mesh checks of `import_mesh(...)` raise a `ValueError` rather than a bare `Exception`, as the same conditions do elsewhere, so one `except ValueError` catches them all.
- `solve(...)` and `Assembly.solve(...)` squeezed every axis of the returned flux and gradient holding a single value, rather than those of the flux alone, so a model of one element lost its element axis and an element with one integration point lost that one.
- `voigt2stiffness(...)` left `C_1122` and `C_2211` at zero in 3D, so converting a stiffness matrix from Voigt notation silently dropped the coupling between the two transverse normal components. It also added a spurious leading dimension to an unbatched input and raised on more than one batch dimension, where the other converters accept any.
- `stiffness2voigt(...)` returned the transpose of the Voigt matrix it builds, which cancelled for the symmetric stiffness tensors it is given but made it no inverse of `voigt2stiffness(...)`.
- `Material.vectorize(...)` dropped a rotation applied before it, because it rebuilt the material from engineering constants that cannot describe a rotated one. `material.rotate(R).vectorize(n)` now gives the same material as `material.vectorize(n).rotate(R)`.
- `planar_rotation(...)`, `axis_rotation(...)` and `euler_rotation(...)` returned the transpose of the matrix they build, so they rotated by `-phi`. They now turn counter-clockwise, i.e. `axis_rotation(...)` follows the right-hand rule and a ply at `+45°` carries its stiff axis at `+45°`. Results that depend on the sign of an angle change: an unbalanced `Laminate` stacking sequence and any material rotated by an angle other than a multiple of 90° are mirrored compared to earlier versions. The orientation markers of `Planar.plot(...)` follow the same convention now.
- Boundary condition glyphs are no longer clipped at the edge of the axes, which their size in points reaches past when the limits are fitted to the mesh.
- `PlanarHeat.plot(...)` draws its thermal boundary conditions: a heat flux as a plus or a minus, and a prescribed temperature as a marker, which was drawn as nothing at all when it was non-zero.
- `time_integration(...)` started from a zero internal flux vector rather than the one belonging to the initial temperature field, which made its first step inconsistent and dropped the trapezoidal rule to first order in time whenever the initial condition was not already an equilibrium. The internal vector it returns at `t=0` is no longer zero.
- `time_integration(...)` added the applied `heat_flux` to the trapezoidal residual instead of subtracting it at both ends of the step, so a transient solve driven by a heat source settled at minus one half of the steady state `solve(...)` gives. Results driven by prescribed temperatures alone are unchanged.
- Fixed a mistake in the triangular shell element from incorrectly implementing an equation from the Krysl paper. `h` is now the element edge length rather than its area, and the element area enters the shear stiffness once instead of twice. Thin-shell results are essentially unchanged, because the two errors cancelled in that limit.

## Version 0.9.0 - August 18 2026

### Added
- `integrate_body_load(...)`, `integrate_surface_load(...)` and `integrate_line_load(...)` turn a distributed load into consistent nodal loads, replacing the lumping that the examples wrote out by hand. Surfaces and lines are picked with a nodal mask, and a float load acts as a pressure along the outward normal.
- `Element.facets` and `Element.facet_type` describe the codimension-1 facets of an element, i.e. the edges of a surface element and the faces of a volume element.
- `FEM.integrate_shape_functions(...)` returns the integral of each shape function over its element, and `FEM.volume_scale` the volume per unit element measure.
- `axes` on `Solid.plot(...)`, `Shell.plot(...)` and `Truss.plot3d(...)` shows labeled coordinate axes, matching the matplotlib plotters.
- `camera` on `Solid.plot(...)`, `Shell.plot(...)` and `Truss.plot3d(...)` sets the camera to a coordinate plane, "iso", or an explicit position, focal point and view up.
- `method="amgx"` solves on the GPU with AmgX algebraic multigrid, an optional backend that needs AmgX built from source and pointed at by `AMGX_DLL`. Iterative solves on CUDA use it automatically once it is installed, needing far fewer iterations than the Jacobi preconditioner. It does not converge on an indefinite tangent, where `method="cg"` or `method="minres"` still do.
- `-method` on `benchmarks/run.py` overrides the linear solver a benchmark problem uses, and every result row now records the backend it actually ran on.
- A fourth benchmark problem `topopt`, mirroring the *structural-mesh* problem of the mosaic benchmark suite: a SIMP cantilever on a random density field, whose 762-fold stiffness spread conditions the system far worse than the uniform fields of the other problems.

### Changed
- GPU support requires CUDA 12 or 13, dropping CUDA 11, whose last CuPy wheel predates the solver signatures used here.
- `sparse_solve(...)` reuses a preconditioner passed as `M` on the GPU instead of rebuilding Jacobi every call, matching the CPU path, so a Newton loop and its adjoint solve share one Jacobi preconditioner.
- `solve(...)` accepts increments that fall as well as rise, so a load cycle like `[0, 1, 0]` unloads instead of silently repeating the state at its peak.
- `integrate_field(...)` is now a contraction of `integrate_shape_functions(...)` and returns the same values as before.
- `solve(...)` no longer builds an element tangent it discards when evaluating the converged state at the end of each increment, which makes `nlgeom=True` and materials with internal state about 10% faster.
- The PyVista plots always show the orientation axes in the corner, which a plain `pyvista.Plotter` skips.
- `cmap` on `Planar.plot(...)`, `Truss.plot2d(...)` and `Truss.plot3d(...)` accepts a matplotlib `Colormap` next to a colormap name, so a truncated or resampled colormap can be passed directly.
- `Element.plot(...)` writes a transparent light and dark figure to `docs/images/shape_functions/`, `<Element>_light.png` and `<Element>_dark.png`, instead of a single opaque one.
- In notebooks, the PyVista plots redraw shortly after loading via `plot_utils.show_html(...)`, so vtk.js does not keep the first frame it draws from its own defaults.
- The `thermal` and `hyperelasticity` benchmarks keep their elements cubic as `N` grows, where a fixed depth used to stretch them into slivers, so both measure problem size rather than element aspect ratio. All published results were re-measured on the new meshes.
- The `hyperelasticity` benchmark picks its solver automatically instead of pinning `cg`, which selects AmgX on CUDA, and starts at `N=35`: the `N=25` case fell below the size at which `resolve_method` turns iterative, so it ran a direct solve slower than every larger case.
- `verbose=True` no longer raises on a console whose codepage lacks the characters the report is drawn with, as a Windows console does by default.

### Fixed
- `method="amgx"` frees each AMG hierarchy when the solve that built it ends, where every increment, cutback and time step used to leave one on the GPU for the life of the process.

## Version 0.8.0 - August 3 2026

### Added
- A PEP 561 `py.typed` marker, so downstream type checkers use the shipped annotations.
- Tests for the conductivity materials, the orthotropic plane-stress and plane-strain materials, `linear_to_quadratic(...)`, `Truss`, `SolidHeat`, the boundary condition setters, and the non-planar mesh imports, none of which had unit tests before.
- A first-order patch test over all eight supported element types. `Planar` and `Solid` were previously only ever tested with `Quad1` and `Hexa1` meshes.
- `torchfem.sparse.resolve_method(...)` returns the linear solver backend that `sparse_solve(...)` picks for a given system size and device. `sparse_solve(...)` now uses it instead of its own copy of the rules.
- `Solid.plot(..., clip=("rho", 0.5))` cuts the mesh at an iso-value of a property, culling orientations and boundary conditions with it. The `optimization/solid/bracket.ipynb` example uses it to show the optimized part instead of exporting a VTU for ParaView.
- `Solid.plot(..., show_outline=True)` draws a box around the full mesh, which gives a clipped result its design space back.
- `Shell.plot(..., plotter=pl)` and `Truss.plot3d(..., plotter=pl)` render into an existing PyVista plotter instead of creating and showing their own, as `Solid.plot(...)` already did.
- `bcs` on `Truss.plot2d(...)` and `Truss.plot3d(...)`, which drew boundary conditions unconditionally. It defaults to True, matching `Planar.plot(...)`, so plots are unchanged unless it is switched off.
- `Shell.plot(..., orientations=...)` draws per-element direction vectors, e.g. the local frames `shell.t`, as red, green, and blue arrows, like `Solid.plot(...)` does. On a shell drawn with `thickness=True` they sit on the top surface instead of inside it. The `optimization/shell/orientation.ipynb` example uses it to show the optimized fiber directions in 3D.
- `Shell.plot(..., show_undeformed=True)` for consistency.
- `Solid.plot(..., orientations=...)` accepts fewer than three vectors per element, like `Shell.plot(...)`, so `[n_elem, 1, 3]` draws a fiber direction alone. The `optimization/solid/topology+orientation.ipynb` example passes just that instead of the full rotated frame.
- Docstrings for `Truss.plot2d(...)` and `Truss.plot3d(...)`, which had none.

### Changed
- The `notebook` extra requires `pyvista[jupyter]` instead of listing `trame`, `trame-vtk`, and `trame-vuetify` itself, so the trame versions stay inside the range PyVista supports. A newer `trame-vtk` ships a VTK.js bundle that PyVista's HTML Jupyter backend embeds incorrectly, leaving the plot blank.
- `Solid.plot(...)` passes `algorithm=None` to `extract_surface(...)`, silencing a PyVista warning about its default changing. The extracted surface is unchanged.
- `Shell.plot(..., mirror=...)` no longer draws the constraint cones that enforce the symmetry of a mirrored plane, since the mirrored copy already shows that symmetry, and warns if the nodes on such a plane are not symmetry-constrained. Loads and all other constraints on the plane are still drawn.
- `Shell.plot(...)` and `Truss.plot3d(...)` no longer set the global PyVista Jupyter backend to `client`. Both pass `jupyter_backend="html"` to `show(...)`, which overrode it anyway, so only the global side effect on other plots is gone.
- `verbose=True` in `solve(...)` and `time_integration(...)` prints a compact table with one row per increment, holding its substeps, iterations, residual, wall time, and flags counting the substep cutbacks (`↓`) and growths (`↑`), under a header naming the model, the machine, the linear solver backend actually used, and the Newton settings. Notebooks redraw the table in place, elsewhere rows stream as they complete. Iterations count linear solves, so a linear problem needs exactly one.
- **Breaking:** The `verbose` argument of `torchfem.sparse.newton_solve(...)` became `report`, taking a `torchfem.report.SolveReport` or `None` instead of a bool.
- Linting and formatting moved from `black`, `isort`, and `flake8` to `ruff`, configured in `pyproject.toml` under `[tool.ruff]`. Local checks are now `ruff format .` and `ruff check --fix .`. Ruff also lints the example notebooks, which the previous stack never covered.
- CI measures test coverage with `pytest-cov` and fails below 78% (currently 81%).
- The `increments` argument of `solve(...)` and the `t_output` argument of `time_integration(...)` now default to `None` instead of a `torch.tensor([0.0, 1.0])` built once at import. The effective default is unchanged.
- The `optimization/solid/bracket.ipynb` example interpolates stiffness as `C_min + rho^p (C0 - C_min)` with `C_min = 1e-3 C0`, instead of `rho^p C0`. The stiffness floor no longer depends on the density bound, so `rho_min` drops from 0.01 to 1e-3 and less of the volume budget is spent on void.
- **Breaking:** `G_13` and `G_23` of `OrthotropicElasticityPlaneStress` and `OrthotropicElasticityPlaneStrain` are unset instead of defaulting to `0.0`, and a homogeneous `Shell(...)` integrates them into its transverse shear stiffness when `transverse_G` is not given. A `Laminate` of plies without transverse moduli silently integrated to zero transverse shear stiffness before and now raises.

### Removed
- **Breaking:** The `contour` argument of `Solid.plot(...)`, superseded by `clip`. The `basic/solid/gyroid.ipynb` example now clips on `thickness - sdf.abs()`, rendering the wall as a solid instead of its two bounding surfaces.
- **Breaking:** The `threshold_condition` argument of `Solid.plot(...)`, superseded by `clip`.
- **Breaking:** The `torchfem.sdfs` module, which provided signed distance functions for implicit geometry (TPMS surfaces, primitives, and CSG booleans) and is out of scope for a finite element library. The `basic/solid/gyroid.ipynb` example now defines its distance function inline.
- The `basic/solid/implicits.ipynb` and `basic/solid/tpms.ipynb` examples and their gallery entries.

### Fixed
- `solve(...)` subdivided every increment of a load path with growing increments. The attempted substep is now carried across increments as a fraction of an increment instead of an absolute size.
- `solve(...)` never recovered to one substep per increment after a cutback. It grew the substep from `step`, which is clipped so the substep lands exactly on the requested increment, instead of from the size it asked for. Every increment therefore ended by shrinking the substep to `growth_factor` times its own last remainder, and kept subdividing ever more finely for the rest of the load path. The `basic/planar/stabilization.ipynb` snap-through now takes 126 Newton iterations instead of 168.
- `__repr__` of `Truss`, `Planar`, `Solid`, and `Shell` reported the element type as `ABCMeta`. It read `self.etype.__class__.__name__`, but `etype` is already a class, so this gave the name of its metaclass.
- `rotate(...)` raised `IndexError` on `OrthotropicElasticityPlaneStrain` and returned meaningless `E_1`, `E_2`, `nu_12`, and `G_12` on `OrthotropicElasticityPlaneStress`. Both inverted the fourth-order stiffness tensor instead of its Voigt matrix, and now use `stiffness2voigt(self.C)` like the 3D class. The rotated `C` was always correct.
- The `IsotropicConductivity3D` docstring documented a non-existent attribute `k` (it is `kappa`) and described `step(...)` as a small-strain elasticity model.
- Several functions shared a single mutable default across all calls: `cached_solve=CachedSolve()` in `sparse_solve(...)`, `newton_solve(...)` and their autograd wrappers, `nodal_data`/`elem_data` in `export_mesh(...)`, and two arguments of `plot_contours(...)`. Each default is now built per call.
- The `psi` function in the `basic/solid/large_compression.ipynb` example read the module-level `mu` and `lbd` instead of its `params`, which would have zeroed gradients with respect to `params`. Results are unchanged.

## Version 0.7.5 - July 30 2026

### Added
- Automatic increment cutback in `solve(...)` via the new `cutback_factor`, `growth_factor`, and `max_cutbacks` arguments. A non-converged Newton solve retries from the last converged state with a smaller substep, and results still come back at exactly the requested `increments`.
- Optional viscous stabilization in `solve(...)` via `alpha`, matching Abaqus automatic stabilization with "Specify damping factor" and disabled by default. `model.stabilization_energy` reports the dissipated energy per increment, equivalent to the Abaqus `ALLSD` output.
- New example `basic/planar/stabilization.ipynb` tracing the snap-through of a shallow cylindrical roof, plus a theory docs section and tests for stabilization.
- `Solid.plot(..., bcs=True)` and `Shell.plot(..., bcs=True)` render boundary conditions: arrows for forces and prescribed displacements, spheres at displacement tips, and a cone per constrained DOF. Shells double the heads for the rotational DOFs, drawing moments as double-headed arrows and constrained rotations as double cones.

### Changed
- The default `max_iter` of `solve(...)` is 10 instead of 100. Exceeding it now triggers an increment cutback rather than aborting the solve.
- `Planar.plot(...)` and `Truss.plot(...)` draw boundary conditions to scale. Force arrows scale with their magnitude instead of all being equally long, prescribed non-zero displacements become an arrow ending in a dot at the position the node is pulled to, and each fixed DOF gets its own marker.
- **Breaking:** Removed the `force_size_factor` and `constraint_size_factor` arguments of `Truss.plot3d(...)`. Markers are sized automatically, cones from the mean bar length.
- 3D trusses draw a sphere at each node smoothing the tube joints, and batch their boundary condition markers into one glyph each, which renders a 343-node truss about 1.7 times faster.
- Type annotations use the builtin generics `tuple`, `list`, and `dict` instead of their deprecated `typing` aliases, and import `Callable` from `collections.abc`. Signatures are unchanged apart from their spelling.

### Fixed
- `Truss.plot(...)` no longer raises when a 3D truss has no applied forces.

## Version 0.7.4 - July 15 2026

### Added
- Regression tests for `time_integration(...)` in `tests/test_time_integration.py`.
- Notebook tests for `basic/solid/thermal_transient.ipynb` and `optimization/planar/orientation_thermal_transient.ipynb`.

### Changed
- **Breaking:** `time_integration(...)` returns one result per requested `t_output` time. Previously `t_output` only set the end time and results came back per internal time step.
- **Breaking:** Removed the `return_intermediate` argument of `time_integration(...)`. Results always carry a leading time axis of length `len(t_output)`; request the internal grid explicitly with `times = torch.arange(0.0, end_time + delta_t, delta_t)`, or index `[-1]` for the final state.
- `time_integration(...)` subdivides each interval between output times into equal substeps of at most `delta_t`, replacing the `torch.arange(...)` and `unique()` merge. Output times are hit exactly, and a requested time close to an internal step no longer adds a spurious near-zero step.
- `time_integration(...)` raises `ValueError` for empty, negative, or non-increasing `t_output`.

### Fixed
- `time_integration(...)` no longer drops the leading time axis of heat flux and temperature gradient when a single output time is requested.

## Version 0.7.3 - July 14 2026

### Added
- Typed mesh import wrappers `import_shell(...)`, `import_planar(...)`, and `import_solid(...)` in `io.py` that return the requested model type and raise `TypeError` when the file's element type does not match.
- New shell size-optimization example `optimization/shell/pressure_vessel.ipynb` (replacing the old `freesize.ipynb`), linked in the docs example gallery and covered by the notebook tests.
- A minimal topology-optimization walkthrough in the getting-started docs, alongside a rewritten planar `topology.ipynb` example.
- Python 3.14 added to the CI test matrix and the PyPI classifiers, plus additional package keywords in `pyproject.toml`.
- Regression test guarding the `differentiable_parameters` trap: `solve()` outputs are fully detached when a grad-requiring design variable is not declared.

### Changed
- `torchfem.sdfs` no longer calls `torch.set_default_dtype(torch.float64)` at import time. SDF constructors now default their tensor arguments to `None` and build them internally, so importing the module no longer mutates the global default dtype.
- `import_mesh(...)` converts mesh points to native-byte-order `float64` before building node tensors (previously cast to `float32`), correctly importing legacy big-endian `.vtk` files and preserving coordinate precision.
- `Shell.plot(...)` forwards `**kwargs` to the underlying PyVista `add_mesh` calls (defaulting to `show_edges=True`), so surface appearance is customizable.

### Fixed
- Forgetting to declare a differentiable argument in a `solve(...)` call now fails loudly: when parameter gradients are not tracked, all outputs are detached instead of silently returning wrong gradients.
- Corrected the performance docs, which still referenced the removed `cubes.ipynb`, and a stale version reference in the publications docs.

## Version 0.7.2 - July 13 2026

### Added
- Two new benchmark problems next to the cube extension: a thermal SIMP slab (mirroring the *thermal-mesh* problem of the mosaic benchmark suite) and a Neo-Hookean large-stretch cube. Both are documented with CPU/GPU results and plots on the performance docs page.
- New example `optimization/solid/source_recovery_thermal.ipynb` recovering a heat source distribution via adjoint optimization, linked in the docs example gallery.
- New cube benchmark results for Apple M1 Pro and RTX 5090.
- Gradient regression tests for multi-increment solves: load-side gradients against a single-step solve, and nonlinear Neo-Hookean material parameter gradients against the analytical uniaxial solution.

### Changed
- Refactored `benchmarks/` into per-problem modules (`cubes.py`, `thermal.py`, `hyperelasticity.py`) sharing a problem interface in `utils.py`. `run.py` runs one or all problems and writes `results/<problem>_<label>.json`. The outdated `cubes.ipynb` is removed.
- `newton_solve(...)` and its `eval_residual` callback now receive the previous increment's state (`u_prev`, `grad_prev`, `flux_prev`, `state_prev`) explicitly.

### Fixed
- Adjoint gradients of multi-increment solves were truncated to the last increment, because the residual closure late-bound the loop variables and the previous state was detached. Gradients now chain across increments and match single-step and analytical references.
- Adapted to CuPy's rename of the CG tolerance argument from `tol` to `rtol`.

## Version 0.7.1 - July 8 2026

### Added
- New optional dependency group `notebook` for running the example notebooks.
- New optional dependency group `dev` with the development tools.
- Binder configuration (`.binder/requirements.txt`) so the Binder badge installs the package with the `notebook` extra.
- CI now enforces linting (`flake8`), formatting (`black`, `isort`), and type checking (`basedpyright`) as dedicated jobs, and the tool configuration lives in `.flake8` and `pyproject.toml`.
- A `notebook` pytest marker so the slow example-notebook tests can be split from the fast unit tests (`pytest -m "not notebook"`).
- A "Models" section in the docs with API documentation for the core model classes (`Truss`, `Planar`, `Shell`, `Solid`, and heat variants) and new docstrings on these classes.
- An examples gallery page in the docs linking all rendered example notebooks.

### Changed
- Split the monolithic `materials.py` into a `torchfem.materials` subpackage (`base`, `elasticity`, `hyperelasticity`, `plasticity`, `damage`, `conductivity`) mirroring the documentation structure. All material classes remain importable from `torchfem.materials`, so existing imports are unaffected.
- Fixed VRAM tracking and updated GPU benchmarks.
- Made torch to cupy handoff in `sparse.py` more memory friendly to reduce VRAM. 
- Slimmed core dependencies: the packages above are only used by the example notebooks.
- Relaxed the SciPy pin from `scipy~=1.15.0` to `scipy>=1.14` and added an explicit `torch>=2.0` lower bound.
- Declared `numpy` as an explicit dependency.
- CI runs the fast unit tests across Python 3.10–3.13 and the notebook tests once, instead of executing every notebook on all four versions.
- Modernize PyPI publishing workflow.
- Restructured the README.
- Complete the theory page in the docs.

### Fixed
- Resolved all `basedpyright` type-checking errors.

## Version 0.7.0 - July 1 2026 

### Added
- Composite laminates for shells via a new `Laminate` section: per-layer material, thickness, and angle, symmetric layups, reference-surface `offset`, per-layer Simpson integration, transverse shear and mass integrals, and nonlinear (state-bearing) plies integrated through the thickness. (Thanks to @yvanblanchard)
- Examples `shell/cantilever_laminate.ipynb`, `shell/cantilever_fml.ipynb` (GLARE fiber-metal laminate), and `shell/copv.ipynb` (composite overwrapped pressure vessel).

### Changed
- `torchfem.data.get_data()` now returns a `pathlib.Path` instead of `str`.

### Fixed
- Plane-stress plasticity: the algorithmic tangent now broadcasts a per-point hardening slope `sigma_f_prime(q)` correctly across a batch.
- Global material `orientation` for `Shell`, projected onto each element to define the ply-angle reference axis (independent of element node ordering).

### Removed
- Unused failure-criteria module.

## Version 0.6.3 - May 18 2026 

### Added
- Alternative text for images in README.md
- Test for example files.
- Test for utils.
- Markdown files to declare contribution, governance, code of conduct, and support.
- Documentation of elements with shape function plots.
- Publications page in docs.
- Theory chapter in docs.
- New publication on C/C-SiC plates added to docs.

### Changed 
- Refactored benchmarks with plots and added them to documentation.
- Improved documentation in truss shape optimization example.
- Integrate shell forces and moments from integration points to enable non-linear materials
- Add basic `shell/plasticity.ipynb` example.
- Significantly enhanced material testing coverage.
- Renamed data helper API from `torchfem.examples.get_example_file(...)` to `torchfem.data.get_data(...)` and moved the module from `examples.py` to `data.py`.
- Skip unnecessary stiffness matrix construction during the backward pass to reduce memory and compute overhead.
- Reduce initial GPU memory peaks with chunked index mapping in `__init__`.

### Removed 
- The utility functions `voigt_strain_rotation` and `voigt_stress_rotation` are not used anywhere. They are removed.
- Dependency on unused `memory_profiler`. This was replaced by a custom profiling function for benchmarking earlier.

### Fixed 
- The FPP example was not working correctly after removing `retain_graph=True` from the sparse solver. Also, detaching `f` in base was introducing an error here. This is now fixed.
- Fixed bug where external strains were validated against `n_nodes` instead of `n_elem` (Thanks to @JulGre).
- Fixed `compute_stiffness` argument not being properly propagated in `integrate_material`.

## Version 0.6.2 - March 25 2026 

### Fixed
- Critical fix for trusses. 

## Version 0.6.1 - March 24 2026 

### Fixed
- Silence warning on sparse invariant checks.
- Fix initial gradient shape in k0 for topology optimization.

## Version 0.6.0 - March 18 2026 

### Added
- Added an adjoint Newton-Raphson autograd operator for nonlinear solves via `newton_solve(...)`.
- Added gradient regression tests in `tests/test_gradients.py` for:
	- consistency between single-step and incremental gradients in mechanics,
	- finite and stable gradients for planar heat topology-style parameters.

### Changed
- Refactored mechanics and heat integration interfaces in `base.py` to use explicit previous-step inputs and return updated integration-point fields instead of mutating global history tensors in-place.
- Renamed the autograd-enabled sparse linear solve entry point from `sparse_solve(...)` to `differentiable_sparse_solve(...)`, while `sparse_solve(...)` now denotes the backend sparse solve routine used by both forward and adjoint paths.
- Updated `solve(...)` and `time_integration(...)` to accept `differentiable_parameters` as either a single tensor or an iterable of tensors.
- Updated nonlinear and transient solve paths to use implicit adjoint logic with cleaner graph handling and optional cached sparse warm starts.
- Improved API and type annotations in solver internals (`sparse.py`, `base.py`) and expanded solver docstrings.
- Expanded differentiability documentation (`docs/differentiability.md`) with explicit sections on:
	- adjoint sparse linear solve,
	- adjoint Newton-Raphson for nonlinear FEM.
- Updated usage examples in `README.md` and `docs/getting_started.md` to pass `differentiable_parameters=...` in differentiable solve calls.
- Updated many notebooks and benchmark scripts/examples to match the current differentiable solve API (single tensor for single-parameter cases, tuple only for multi-parameter cases).
- Accelerated assembly by precomputing sparsity patterns (notably helping iterative optimization examples).
- Added meshio compression toggle support.

### Fixed
- Ensured plotting utilities move tensors to CPU before plotting to avoid backend/device issues.

## Version 0.5.1 - January 14 2026 

### Added 
- Added a new example "property_fields.ipynb" for neural fields in the planar optimization examples.

### Changes 
- In the backward sparse solve, we solve the adjoint problem with A_T. Since A is symmetric, we can use the exact same preconditioner M from the forward pass again in iterative methods. This saves us the overhead of creating the preconditioner again and accelerates backward passes massively. 
- Improve construction of sparse gradient in adjoint backward path of the sparse solver knowing that it is coalesced. 
- Vectorize material parameters in hyperelastic materials
- Vectorize evaluation of shape function. This accelerates in particular frequent small solves in inverse problems.
- Make characteristic length cached properties to prevent frequent recomputation in inverse problems.


## Version 0.5.0 - December 19 2025 

### Added 
- Added `PlanarHeat` and `SolidHeat` for heat transfer problems (Thanks to @kraussco).
- Added new planar examples "thermal_static.ipynb", "thermal_transient.ipynb", "orientation_thermal_static.ipynb", "topology_thermal_static.ipynb" for heat transfer and thermal optimization (Thanks to @kraussco).
- Added new solid examples "thermal_static.ipynb", "thermal_transient.ipynb", "topology_thermal.ipynb" for heat transfer and thermal optimization (Thanks to @kraussco).
- Added export of animated results.
- Better meshing capabilities in the `mesh` module (structured tet meshes, structured tri meshes) to remove dependency on meshzoo. 
- Warning message for single precision solves.
- Added new example geometry (*.vtu) of a quarter symmetric plate.
- Added two new solid examples "isotropic_damage.ipynb" and "plate_damage.ipynb"
- Add simple damage model 'IsotropicDamage3D'.
- \_\_repr\_\_ functions to print torch-fem objects.


### Changed 
- Split the base FEM class into a `Mechanics` and a `Heat` class with generic fluxes.
- Shells are now properly integrated in the parent classes by inheriting from `Mechanics`.
- Simplified thickness assignments for planar and shell meshes.
- Planar plots show vectors, if the provided property is multi-dimensional.
- Material 'step' functions get an additional input 'cl' for the characteristic length of each element. This can be used for regularization in damage models.
- Accelerate element potting for planar models.
- Accelerate 'linear_to_quadratic()' function for elements.
- Accelerate filter matrix H in 'bracket.ipynb' topology optimization example with KD Tree.
- Planar plot uses explicit triangulation objects.
- Truss plot accepts u as positional argument to match base class.

### Fixed
- Fixed some typing issues.
- In some cases the planar contour plot did not show the highest contour level correctly. This is fixed now.
- The hyperelasticity was somewhat working, but not strictly correct and failed to converge at very large strains. Now, we use a Total Lagrangian Formulation, which is robustly and (hopefully) correctly implemented.
- Corrected type hints in `export_mesh` for elem_data.
- The size of the stiffness tensor for `OrthotropicElasticityPlaneStrain` was incorrect. It is corrected from (3,3,3,3) to (2,2,2,2).

### Removed 
- The material classes `IsotropicHencky3D`, `IsotropicHenckyPlanarStrain` and `IsotropicHenckyPlanarStress` are removed. Use the more general hyperelastic models instead.
- Dependency on meshzoo. This was limited to a few nodes with a license - use the internal functions in the `mesh` module instead.

## Version 0.4.5 - June 05 2025 

### Changed
- Instead of specifying 'direct=True in the 'fem.solve' function, you can now specify 'method=["cg", "minres", "spsolve", "pardiso"]' for more fine-grained control over the solver selection. 

## Version 0.4.4 - June 04 2025 

### Changed
- The 'NeoHookean3D' material model has been replaced by a general 'Hyperelastic3D' model. This accepts an energy function depending on the right Cauchy Green tensor and computes Cauchy stress and the spatial material tangent using automatic differentiation.
- Example 'basic/planar/large_stretch' is updated to use the new 'Hyperelastic3D' model.
- Example 'basic/planar/rubber_stretch' is updated to use the new 'Hyperelastic3D' model.


## Version 0.4.3 - April 04 2025 

### Changed
- Installation with CUDA is explained in more detail (Addressing #23).

### Fixed
- Solver option `device="cuda"` works properly now (Fixing #20).


## Version 0.4.0 - April 04 2025 

### Added 
- This CHANGELOG.md
- Example 'basic/planar/finite_strain' with a hole plate subjected to finite strains. 
- Example 'basic/planar/large_stretch' with a rectangle subjected to a large principal stretch 'λ=5'.
- Example 'basic/solid/finite_strain' comparing a cantilever beam subjected to loading at its tip with and without nonlinear geometry. 
- Example 'basic/solid/large_stretch' with a cube subjected to a large principal stretch 'λ=5'.
- Example 'basic/solid/rubber_stretch' for a ISO37 rubber specimen with a Neo-Hookean material.
- Example 'optimization/solid/topology+orientation' for concurrent topology and orientation optimization in 3D. 
- New functions to convert between Voigt notation and tensor notation in 'utils.py'
- New materials ('IsotropicHencky3D', 'IsotropicHenckyPlaneStrain', 'IsotropicHenckyPlaneStress', 'NeoHookean3D')
- New function 'rect_quad' in 'mesh.py' as 2D variant of 'cube_hex'.
- Orientation option in solid plot.

### Changed
- **Instead of Voigt notation, we use full tensor notation for all stresses, strains, stiffnesses etc.** 
- FEM 'forces', 'displacements', and 'constraints' are now set via attribute setters to detect mistakes (Thanks @aeverallpx)
- The solver 'FEM.solve(...)' returns now a deformation gradient 'F' instead of infinitesimal strain 'ε'.  
- The default for 'max_iter', i.e. the maximum number of Newton-Raphson iterations in the solver is increased to 100.
- The solver 'FEM.solve(...)' accepts an additional boolean 'nlgeom' to indicate wether it should account for geometric non-linearity. If set to yes, the shape functions are evaluated on the deformed configuration (updated Lagrangian) and the stiffness also accounts for geometric stiffness.
- Instead of 'FEM.D(...)' we now have 'FEM.eval_shape_functions()' as abstract method.
- The material functions 'Material.step(...)' changed its arguments. The first argument is now a displacement gradient increment instead of the elastic infinitesimal strain increment. The second argument is the current deformation gradient and not the current infinitesimal strain. Inelastic external strains increments (such as thermal strains) are passed in as additional argument 'de0'. 
- Better documentation of materials. 

### Fixed
- K is now coalesced once more at the end of assembly to fix wrong gradients on GPU (Thanks @aeverallpx)
- Colorbar option in planar plot.
