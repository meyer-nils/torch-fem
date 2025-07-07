# Changelog 

## Unreleased

### Added 
- \_\_repr\_\_ functions to print torch-fem objects.

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
