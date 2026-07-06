---
icon: lucide/cog
---

# Installation
You may install torch-fem via pip with
``` sh
pip install torch-fem
```

## Example notebooks (optional)
The core package ships only with the dependencies required to build and solve models. To run the example notebooks, install with the `notebook` extra to include Jupyter widgets, interactive PyVista rendering, and animation support:
``` sh
pip install torch-fem[notebook]
```

## GPU support (optional)
For optional GPU support, install CUDA, PyTorch for CUDA, and the corresponding CuPy version.

For CUDA 11.8: 
``` sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x # v11.2 - 11.8
```

For CUDA 12.9:
``` sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install cupy-cuda12x # v12.x
```


## Development (optional)
To develop new features for *torch-fem*, you should fork the GitHub repository and clone it to your machine via 
``` sh 
git clone <repo_url> 
```

In the root of the cloned project, make the installation editable as 
``` sh 
pip install -e ".[all]"
```
The `all` extra combines the `notebook` extra with the development tools (`pytest`, `testbook`, `flake8`) in the `dev` extra. Both are required to run the test suite, which executes the example notebooks.

Now, the package is linked to this local directory and whenever you use `import torchfem`, it will use the latest code.