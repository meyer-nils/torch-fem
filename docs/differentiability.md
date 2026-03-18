---
icon: lucide/wand-sparkles
---

# Differentiability

*torch-fem* is called *torch-fem*, because it is built on top of *PyTorch* - one of the most popular frameworks for machine learning applications.

This allows us to compute gradients easily and let them flow seamlessly through the FEM Solver - think of torch-fem as just another layer in your ML pipeline. This opens many opportunities:

- Combine Neural Networks and numerical solvers to train them together.
- Solving optimization tasks such as topology optimization, optimization of fiber orientations, or parameter identification.
- Formulate Hyperelastic materials by simply specifying the strain energy density function.

## How differentiability is achieved

At its core, *torch-fem* keeps the full FEM pipeline in tensor form and uses `torch` operations wherever possible (assembly, constitutive updates, residual evaluation, loss construction). This guarantees that all local operations participate in PyTorch autograd and can be differentiated with `.backward()`.

For the global sparse linear algebra and nonlinear equilibrium solves, *torch-fem* uses custom autograd operators with adjoint equations instead of unrolling all solver iterations in the computational graph.

### Adjoint sparse linear solve

For a linear system
$$
A(\theta)x = b(\theta),
$$
the forward pass computes $x$ with a non-differentiable sparse backend (for example SciPy, CuPy, or Pardiso). In the backward pass, given an upstream gradient $\partial \mathcal{L}/\partial x$, we solve the adjoint system
$$
A^\top\lambda = \partial \mathcal{L}/\partial x.
$$
Then sensitivities follow from
$$
\frac{\partial \mathcal{L}}{\partial b} = \lambda,
\qquad
\frac{\partial \mathcal{L}}{\partial A} = -\lambda x^\top
$$
(restricted to the stored sparse entries).

This means gradients are obtained by one additional sparse solve, not by differentiating through each internal solver iteration.

### Adjoint Newton-Raphson for nonlinear FEM

For nonlinear equilibrium, we solve at convergence
$$
R(u, \theta) = 0,
\qquad
K = \frac{\partial R}{\partial u}.
$$
Instead of differentiating through all Newton updates, we apply the implicit/adjoint relation at the converged state $u^*$. For a scalar objective $\mathcal{L}(u,\theta)$:
$$
K^\top\lambda = \frac{\partial \mathcal{L}}{\partial u^*},
$$
and
$$
\frac{d\mathcal{L}}{d\theta}
=
\frac{\partial \mathcal{L}}{\partial \theta}
-
\lambda^\top\frac{\partial R}{\partial \theta}.
$$

Practically, this has two major benefits:

- Memory and runtime scale much better than backpropagating through every Newton and linear-solver iteration.
- The forward solver can use non-differentiable external sparse backends (SciPy/CuPy/Pardiso), while gradients remain correct through the adjoint equations.

In other words, the differentiable interface is preserved even when the numerical linear algebra backend itself is not natively differentiable.



## Neural field example

Let's assume we have a graded property field, where the Young's modulus is prescribed as a function $E : \mathbb{R}^2 \rightarrow \mathbb{R}$ on the domain $\Omega \in [0, 2] \times [0, 1]$.

The function is given by
$$
    E(x) = 2 + \sin(\pi x_0) \cos(\pi x_1)
$$

We solve the FEM problem with this exact field to get a synthetic ground truth of the deformation $u_\textrm{ref}$:

![Deformation of the structure with a property field](images/property_field.png)

### Direct solution of the inverse problem

Let's solve the inverse problem now, i.e, we want to estimate the property field leading to the observed deformation field $u_\textrm{ref}$. Hence, we solve

$$
    \min_E \left( u(E) - u_\textrm{ref} +  u_\textrm{noise} \right)^2
$$

with the computed deformation field $u$ given the modulus field $E$ on elements and some observation noise $u_\textrm{noise}$. We solve this problem directly by taking the modulus of each element as the design variable and performing a gradient descent with Adam. Note that this is only possible, because we can call `.backward()` on our loss function and compute the gradients via automatic differentiation.

![Direct reconstruction of the property field](images/property_field_direct.png)

### Solution of the inverse problem with a neural field

The previous attempt recovers the discrete distribution of the elastic modulus. However, it does not recover the continuous underlying function and is susceptible to noise. Therefore, we introduce a neural field as approximation to the continuous stiffness distribution, which is trained through the FE solver with noisy reference displacements.

The stiffness must be positive, in particular we know
$$
E(x) > 1 \quad \forall x \in \Omega
$$
a priori in this task. Therefore, we enforce this physical property in the NN design with a ReLU output activation layer and addition of a constant in the output layer.

Training the neural field takes more iterations in order to get to a comparable accuracy. However, the neural field regularizes the property field and gives a much smoother representation that is less susceptible to noise. This training strategy is only possible because gradients can flow seamlessly between FEM and neural network.

![Reconstruction of the property field with a neural field](images/property_field_neural.png)




[View example on GitHub :fontawesome-brands-github:](https://github.com/meyer-nils/torch-fem/blob/main/examples/optimization/planar/property_fields.ipynb){ .md-button }
[Open in Google Colab :fontawesome-brands-google-drive:](https://colab.research.google.com/github/meyer-nils/torch-fem/blob/main/examples/optimization/planar/property_fields.ipynb){ .md-button }
