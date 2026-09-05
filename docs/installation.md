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
Install PyTorch for your CUDA version. For CUDA 13:
``` sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

For CUDA 12, use the `cu129` index URL instead. CUDA 11 is not supported.

## Advanced solver backends (optional)
*torch-fem* runs its iterative solvers in PyTorch itself, on either device, and falls back to SciPy for a direct solve and for the algebraic multigrid preconditioner on CPU. Two optional backends can be faster, and *torch-fem* picks up either one as soon as it is installed.

**Pardiso** is a direct CPU solver:
``` sh
pip install pypardiso
```
Once installed, *torch-fem* uses it automatically for a direct solve, which is what small models get by default.

**AmgX** is NVIDIA's GPU algebraic multigrid library, which needs far fewer iterations than a Jacobi preconditioner. It ships no wheels, so build it from source against a matching CUDA toolkit by following the instructions in the [AmgX repository](https://github.com/NVIDIA/AMGX), then point *torch-fem* at the resulting shared library:
``` sh
export AMGX_DLL=/path/to/libamgxsh.so # amgxsh.dll on Windows
```
Iterative solves on the GPU then use it automatically. Pass `preconditioner="jacobi"` to fall back to the diagonal one.

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
