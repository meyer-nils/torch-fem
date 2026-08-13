---
icon: lucide/gauge
---

# Performance

Scaling behavior of *torch-fem* on three benchmark problems across several machines. The plotted time is the total of three phases, each of which is recorded separately in the [result files](https://github.com/meyer-nils/torch-fem/tree/main/benchmarks/results):

1. **Setup:** mostly computing the sparsity pattern.

2. **Forward solve:** assembly and sparse linear solve.

3. **Backward solve:** reverse-mode AD through the solve via `autograd`.

## Cube extension

<figure markdown="span">
![Cube extension, displacement magnitude](images/benchmark/cube_model_light.png#only-light){ width="420" }
![Cube extension, displacement magnitude](images/benchmark/cube_model_dark.png#only-dark){ width="420" }
</figure>

A unit cube from linear hexahedra with $N$ nodes along each edge ($3N^3$ degrees of freedom) in isotropic linear elasticity ($E = 1000$, $\nu = 0.3$), clamped at $x = 0$ and pulled to $u_x = 0.1$ at $x = 1$. The backward pass takes the gradient of `u.sum()` with respect to the nodal forces.

![Total time scaling](images/benchmark/cube_timing_light.png#only-light)
![Total time scaling](images/benchmark/cube_timing_dark.png#only-dark)

![Peak RAM scaling](images/benchmark/cube_ram_light.png#only-light)
![Peak RAM scaling](images/benchmark/cube_ram_dark.png#only-dark)

## Thermal SIMP slab

<figure markdown="span">
![Heated slab, temperature field](images/benchmark/thermal_model_light.png#only-light){ width="420" }
![Heated slab, temperature field](images/benchmark/thermal_model_dark.png#only-dark){ width="420" }
</figure>

A quasi-2D slab on $[0,2] \times [0,1]$, one layer of cubic hexahedra deep and $N$ elements along the long edge, carrying SIMP-penalized conductivity $k(\rho) = k_\text{min} + (k_\text{max} - k_\text{min})\rho^3$ at uniform $\rho = 0.5$. It is cold at $x = 0$ and heated by a uniform flux at $x = 2$; the backward pass is the adjoint sensitivity of the thermal compliance with respect to the per-element densities. This mirrors the *thermal-mesh* problem of the [mosaic benchmark suite](https://github.com/pasteurlabs/mosaic).

![Total time scaling](images/benchmark/thermal_timing_light.png#only-light)
![Total time scaling](images/benchmark/thermal_timing_dark.png#only-dark)

![Peak RAM scaling](images/benchmark/thermal_ram_light.png#only-light)
![Peak RAM scaling](images/benchmark/thermal_ram_dark.png#only-dark)

## Neo-Hookean stretch

<figure markdown="span">
![Neo-Hookean stretch, displacement magnitude at a 2x stretch](images/benchmark/hyperelasticity_model_light.png#only-light){ width="420" }
![Neo-Hookean stretch, displacement magnitude at a 2x stretch](images/benchmark/hyperelasticity_model_dark.png#only-dark){ width="420" }
</figure>

A box of Neo-Hookean material stretched to ten times its length in 10 increments, geometric in the stretch, with full Newton iterations (`nlgeom=True`), mirroring the [large stretch example](https://github.com/meyer-nils/torch-fem/blob/main/examples/basic/solid/large_stretch.ipynb). Only $y$ and $z$ are refined, with $N$ nodes each over four cubic elements along the stretch direction ($15N^2$ degrees of freedom), since the solution is homogeneous and a longer stretch drives the tangent indefinite. The forward solution matches the analytical uniaxial response; the backward pass is the adjoint of the total reaction force with respect to the Lamé parameters, as used in material calibration.

![Total time scaling](images/benchmark/hyperelasticity_timing_light.png#only-light)
![Total time scaling](images/benchmark/hyperelasticity_timing_dark.png#only-dark)

![Peak RAM scaling](images/benchmark/hyperelasticity_ram_light.png#only-light)
![Peak RAM scaling](images/benchmark/hyperelasticity_ram_dark.png#only-dark)

## Reproducing the results

The scripts live in `benchmarks/` at the repository root.

**1. Run the benchmark**:

```bash
# All benchmarks on CPU (default)
python benchmarks/run.py

# Structural benchmark on CUDA
python benchmarks/run.py -problem cube -device cuda --label rtx5090_cuda --hardware "RTX 5090"

# Thermal benchmark
python benchmarks/run.py -problem thermal --label m1_pro_cpu --hardware "Apple M1 Pro"

```

The label identifies the machine; results are written to `benchmarks/results/<problem>_<label>.json`.

**2. Regenerate the plots**:

```bash
python benchmarks/plot.py
```

This reads all JSON files in `benchmarks/results/`, groups them by problem, and writes the timing, backward, and RAM plots to `docs/images/benchmark/<problem>_*.png`. 