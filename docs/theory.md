---
icon: lucide/pi
---

# Theory

This page summarizes the continuum mechanics and finite-element formulations implemented in *torch-fem*. The implementation supports solving mechanical problems with differentiable solution operators built on PyTorch autograd.

## Mechanics

### Kinematics

Let $\Omega \subset \mathbb{R}^d$ denote the reference domain with material point $\mathbf{X}$ and displacement field $\mathbf{u}$. The deformed configuration is

$$
\mathbf{x} = \mathbf{X} + \mathbf{u},
$$

and the deformation gradient is

$$
\mathbf{F} = \nabla \mathbf{x} = \mathbf{I} + \nabla \mathbf{u} = \mathbf{I} + \mathbf{H}
$$

with the displacement gradient $\mathbf{H} = \nabla \mathbf{u}$, where the operator $\nabla \mathbf{u} = \frac{\partial u_i}{\partial X_j}$ denotes the gradient w.r.t. the reference configuration.

### Momentum balance

The fundamental equation to solve in static structural problems is the balance of linear momentum. In quasi-static settings (neglecting inertia) and without body forces (neglecting gravity), it reads

$$
\mathrm{div} \left( \mathbf{P} \right) = \mathbf{0} \quad \mathbf{X} \in \Omega,
$$

where $\mathbf{P}$ is the first Piola-Kirchhoff stress.

The boundary $\partial \Omega = \partial \Omega_E \cup \partial \Omega_N$ with $\partial \Omega_E \cap \partial \Omega_N = \emptyset$ is split into an *essential* boundary $\partial \Omega_E$, where displacements $\mathbf{u}_0$ are prescribed, and a *natural* boundary $\partial \Omega_N$, where tractions $\mathbf{t}$ are prescribed. The boundary value problem then reads

$$
\begin{aligned}
    \mathrm{div} \left( \mathbf{P} \right) &= \mathbf{0}
        & \mathbf{X} &\in \Omega \\
    \mathbf{u} &= \mathbf{u}_0 
        & \mathbf{X} &\in \partial \Omega_E \\
    \mathbf{P} \cdot \mathbf{n} &= \mathbf{t}
        & \mathbf{X} &\in \partial \Omega_N
\end{aligned}
$$

with the outward unit normal $\mathbf{n}$. This is a second-order problem for the unknown displacement field $\mathbf{u}$.

### Material models

Material models describe how the stress in a material depends on its deformation. In a general sense, we can formulate this for an isothermal material abstractly as

$$
 \mathbf{P} = \mathcal{F}(\mathbf{F}, \pmb{\alpha})
$$

with a function $\mathcal{F}$ that maps a deformation gradient $\mathbf{F}$ and material state vector $\pmb{\alpha}$ to a stress.

!!! info

    *torch-fem* implements this relation incrementally in the `step()` method of each material, which also provides the algorithmic tangent used by the solver. See [Materials](materials/index.md) for details and all available models.

## Finite element method

### Weak form

To solve the *strong form* of the problem above, the displacement field has to be twice differentiable. We can relax that requirement by computing the *weak form*: we contract the equation with a test function $\mathbf{v} \in \mathcal{H}^1_0 (\Omega)$, i.e. a function that is zero at the essential boundary and has square integrable derivatives, integrate over the domain $\Omega$, and apply the divergence theorem. Using $\mathbf{v} = \mathbf{0}$ on $\partial \Omega_E$ and $\mathbf{P} \cdot \mathbf{n} = \mathbf{t}$ on $\partial \Omega_N$, the result is

$$
\int_{\partial \Omega_N} \mathbf{t} \cdot \mathbf{v} \,\mathrm{d}A
- \int_\Omega \mathbf{P} : \nabla \mathbf{v} \,\mathrm{d}V = 0
\quad \forall \mathbf{v}.
$$

This is equivalent to the strong form as long as we require the equation to hold for arbitrary $\mathbf{v}$, and it may be interpreted as the principle of virtual work with the test function taking the role of a virtual displacement. The benefit of this formulation is that the solution $\mathbf{u}$ has to be only differentiable once.

### Discretization

In general, we cannot solve the weak form analytically on complex domains. The core idea of finite elements is the discretization of the continuous domain $\Omega$ into a finite number of $M$ small elements $\Omega_j$, on which the solution can be approximated with simple functions. The weak form then becomes a sum over all elements

$$
\sum_{j=1}^{M} \int_{\partial \Omega_{N,j}} \mathbf{t} \cdot \mathbf{v} \,\mathrm{d}A
- \sum_{j=1}^{M} \int_{\Omega_j} \mathbf{P} : \nabla \mathbf{v} \,\mathrm{d}V = 0
\quad \forall \mathbf{v}.
$$

### Shape functions

Within each element $\Omega_j$, the solution is approximated with *shape functions*. These are simple polynomial functions that are continuous across element edges and defined on a *reference element* with local coordinates $\pmb{\xi}$. Any variable defined at the element nodes $\mathbf{a}_j$ can be interpolated in the element as

$$
a(\pmb{\xi}) = \mathbf{N}(\pmb{\xi}) \cdot \mathbf{a}_j
$$

with shape functions $\mathbf{N}(\pmb{\xi})$ that assume the value 1 at one node of the element and the value 0 at all other nodes. Their gradients w.r.t. the local coordinates

$$
\mathbf{B}^\textrm{ref}(\pmb{\xi}) = \nabla_\xi \mathbf{N}(\pmb{\xi})
$$

allow us to compute gradients of interpolated variables on the reference element. The same shape functions are used to interpolate the nodal positions $\mathbf{X}_j$ of an element (*isoparametric concept*). This defines the *Jacobian*

$$
\mathbf{J}_j(\pmb{\xi}) = \frac{\partial \mathbf{X}}{\partial \pmb{\xi}}
= \mathbf{B}^\textrm{ref}(\pmb{\xi}) \cdot \mathbf{X}_j,
$$

which maps between local and global coordinates. Employing the chain rule, gradients w.r.t. the global coordinates are

$$
\nabla a = \underbrace{\mathbf{J}_j^{-1}(\pmb{\xi}) \cdot \mathbf{B}^\textrm{ref}(\pmb{\xi})}_{\mathbf{B}(\pmb{\xi})} \cdot \mathbf{a}_j.
$$

Integrals over an element are evaluated numerically with *Gauss-Legendre quadrature* using tabulated positions $\pmb{\xi}^k$ and weights $w^k$ on the reference element:

$$
\int_{\Omega_j} f \,\mathrm{d}V
= \sum_{k} w^k \, \textrm{det}\left(\mathbf{J}_j(\pmb{\xi}^k)\right) f(\pmb{\xi}^k).
$$

The determinant of the Jacobian acts as a scaling factor between the reference element and the actual element.

!!! info

    In *torch-fem*, each element formulation provides exactly these ingredients: shape functions via `N()`, local derivatives via `B()`, and quadrature positions and weights via `ipoints` and `iweights`. See [Elements](elements/index.md) for all available element types and plots of their shape functions.

### Element matrices

Interpolating both $\mathbf{u}$ and $\mathbf{v}$ with shape functions turns the element integrals into sums over integration points. Denoting the gradient of the shape function of node $I$ by $\nabla N_I$, i.e. the $I$-th column of $\mathbf{B}(\pmb{\xi}^k)$, quadrature yields the internal nodal forces and the tangent stiffness of an element

$$
\left(\mathbf{f}_j\right)_{aI} = \sum_k w^k \, \textrm{det}\left(\mathbf{J}_j(\pmb{\xi}^k)\right) P_{ai} \frac{\partial N_I}{\partial X_i}
\quad \textrm{and} \quad
\left(\mathbf{k}_j\right)_{aIbJ} = \sum_k w^k \, \textrm{det}\left(\mathbf{J}_j(\pmb{\xi}^k)\right) \frac{\partial N_I}{\partial X_i} \, \mathbb{C}_{aibl} \, \frac{\partial N_J}{\partial X_l},
$$

where the stress $\mathbf{P}$ and the algorithmic tangent $\mathbb{C}$ are provided by the material model at each integration point. *torch-fem* evaluates these expressions directly as batched tensor contractions of $\mathbf{B}$, the stress, and the tangent over all elements at once. For planar models, the contributions are additionally scaled with the element thickness $d_j$, and for trusses with cross-sectional areas.

The boundary traction term reduces analogously to consistent nodal forces $\mathbf{f}^\Gamma_j$. In *torch-fem*, such nodal forces are prescribed directly via `model.forces`, and prescribed displacements $\mathbf{u}_0$ are set via `model.displacements` and `model.constraints`.

### Assembly

Summing all element contributions and using the fact that the test function $\mathbf{v}$ is arbitrary yields a global equation system

$$
\mathbf{K} \cdot \mathbf{u} = \mathbf{f},
$$

which is assembled by adding the entries of each element stiffness matrix $\mathbf{k}_j$ at the positions of the corresponding global degrees of freedom. In *torch-fem*, this assembly is a vectorized scatter operation into a sparse tensor with a precomputed sparsity pattern.

Degrees of freedom constrained on the essential boundary remain part of this system: their values are set directly to the prescribed displacements, and their equations are replaced by trivial identities by zeroing the corresponding stiffness entries within the fixed sparsity pattern and placing ones on the diagonal.

### Solution procedure

In the general nonlinear case, the discretized problem is a root-finding problem for the residual

$$
\mathbf{R}(\mathbf{u}) = \mathbf{f}^\textrm{int}(\mathbf{u}) - \mathbf{f}^\textrm{ext} = \mathbf{0}.
$$

The external loads are applied in load increments $0 = \lambda_0 < \lambda_1 < \dots < \lambda_N = 1$ and each increment is solved with a Newton-Raphson method: in every iteration, the linearized system

$$
\mathbf{K} \cdot \Delta \mathbf{u} = -\mathbf{R}(\mathbf{u})
$$

is solved with the tangent stiffness $\mathbf{K}$ assembled from the algorithmic material tangents. For linear problems, this converges in a single iteration.

The sparse linear system can be solved with different backends via the `method` argument of `solve()`:

- direct sparse solvers (`"spsolve"`, `"pardiso"`),
- iterative Krylov solvers (`"cg"`, `"minres"`) with an algebraic multigrid preconditioner built from rigid-body modes.

On CUDA devices, the sparse solves are performed with CuPy for GPU acceleration.

!!! info

    All solution operators in *torch-fem* are differentiable: instead of differentiating through individual Newton or solver iterations, gradients are computed with adjoint methods derived from the implicit function theorem. See [Differentiability](differentiability.md) for details.
