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

A model and its solves run on the GPU once it is the default device:
``` py
torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")
```

## Advanced solver backends (optional)
*torch-fem* runs its iterative solvers in PyTorch itself, on either device, with an algebraic multigrid preconditioner built by pyamg on CPU, and falls back to SciPy for a direct solve. Two optional backends can be faster, and *torch-fem* picks up either one as soon as it is installed.

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
The `all` extra combines the `notebook`, `dev` and `docs` extras. The unit tests need `dev` alone, while the notebook tests execute the examples and need `notebook` too.

Now, the package is linked to this local directory and whenever you use `import torchfem`, it will use the latest code.
