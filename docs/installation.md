---
icon: lucide/cog
---

# Installation
Install *torch-fem* with the `notebook` extra to get Jupyter widgets, interactive PyVista rendering, and animation support, which the example notebooks need:
``` sh
pip install torch-fem[notebook]
```

Plain `pip install torch-fem` installs only what is required to build and solve models, if you do not intend to run the notebooks.

## GPU support (optional)
Install PyTorch and CuPy for your CUDA version. For CUDA 13:
``` sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install cupy-cuda13x
```

For CUDA 12, use the `cu129` index URL and `cupy-cuda12x` instead. CUDA 11 is not supported.

## Advanced solver backends (optional)
*torch-fem* solves with SciPy on CPU and CuPy on GPU out of the box. Two optional backends can be faster, and both are selected per solve with `method=...` in `solve(...)`.

**Pardiso** is a direct CPU solver:
``` sh
pip install pypardiso
```
Once installed, *torch-fem* uses it automatically for small models on CPU, where a direct solve beats an iterative one. Pass `method="pardiso"` to force it.

**AmgX** is NVIDIA's GPU algebraic multigrid library, which takes roughly 2.5x fewer iterations than the Jacobi preconditioner of the GPU default. It ships no wheels, so build it from source against a matching CUDA toolkit by following the instructions in the [AmgX repository](https://github.com/NVIDIA/AMGX), then point *torch-fem* at the resulting shared library:
``` sh
export AMGX_DLL=/path/to/libamgxsh.so # amgxsh.dll on Windows
```
Pass `method="amgx"` to use it. It is never selected automatically.

## Development (optional)
To develop new features for *torch-fem*, you should fork the GitHub repository and clone it to your machine via 
``` sh 
git clone <repo_url> 
```

In the root of the cloned project, make the installation editable as 
``` sh 
pip install -e ".[all]"
```
The `all` extra combines the `notebook` extra with the development tools (`pytest`, `testbook`, `ruff`) in the `dev` extra. Both are required to run the test suite, which executes the example notebooks.

Now, the package is linked to this local directory and whenever you use `import torchfem`, it will use the latest code.
