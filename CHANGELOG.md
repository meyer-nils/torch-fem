# Changelog 

## Unreleased

### Added
- New optional dependency group `notebook` for running the example notebooks.
- New optional dependency group `dev` with the development tools.
- Binder configuration (`.binder/requirements.txt`) so the Binder badge installs the package with the `notebook` extra.
- CI now enforces linting (`flake8`), formatting (`black`, `isort`), and type checking (`basedpyright`) as dedicated jobs, and the tool configuration lives in `.flake8` and `pyproject.toml`.
- A `notebook` pytest marker so the slow example-notebook tests can be split from the fast unit tests (`pytest -m "not notebook"`).

### Changed
- Split the monolithic `materials.py` into a `torchfem.materials` subpackage (`base`, `elasticity`, `hyperelasticity`, `plasticity`, `damage`, `conductivity`) mirroring the documentation structure. All material classes remain importable from `torchfem.materials`, so existing imports are unaffected.
- Fixed VRAM tracking and updated GPU benchmarks.
- Made torch to cupy handoff in `sparse.py` more memory friendly to reduce VRAM. 
- Slimmed core dependencies: the packages above are only used by the example notebooks.
- Relaxed the SciPy pin from `scipy~=1.15.0` to `scipy>=1.14` and added an explicit `torch>=2.0` lower bound.
- Declared `numpy` as an explicit dependency.
- CI runs the fast unit tests across Python 3.10–3.13 and the notebook tests once, instead of executing every notebook on all four versions.
- Modernize PyPI publishing workflow.

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
