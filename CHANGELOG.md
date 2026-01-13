# Changelog 

## Unreleased

### Added 
- Added a new example "property_fields.ipynb" for neural fields in the planar optimization examples.

### Changes 
- In the backward sparse solve, we solve the adjoint problem with A_T. Since A is symmetric, we can use the exact same preconditioner M from the forward pass again in iterative methods. This saves us the overhead of creating the preconditioner again and accelerates backward passes massively. 
- Vectorize material parameters in hyperelastic materials


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
