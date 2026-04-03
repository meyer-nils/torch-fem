---
icon: lucide/boxes
---

# Elements

Element formulations in `torch-fem` define interpolation on a reference domain and quadrature rules for numerical integration.

For an element with nodal coordinates $\mathbf{x}_i$ and shape functions $N_i(\pmb{\xi})$, the isoparametric mapping is

$$
\mathbf{x}(\pmb{\xi}) = \sum_{i=1}^{n} N_i(\pmb{\xi})\,\mathbf{x}_i.
$$

All concrete elements derive from the abstract `Element` base class and provide

- shape functions via `N()`
- derivatives in reference coordinates via `B()`
- reference nodal coordinates via `iso_coords`
- integration points and weights via `ipoints` and `iweights`