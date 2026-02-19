---
icon: lucide/cog
---

# Installation
You may install torch-fem via pip with
``` sh
pip install torch-fem
```

## GPU support (optional)
For optional GPU support, install CUDA, PyTorch for CUDA, and the corresponding CuPy version.

For CUDA 11.8: 
``` sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x # v11.2 - 11.8
```

For CUDA 12.6:
``` sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install cupy-cuda12x # v12.x
```


## Development (optional)
To develop new features for *torch-fem*, you should fork the GitHub repository and clone it to your machine via 
``` sh 
git clone <repo_url> 
```

In the root of the cloned project, make the installation editable as 
``` sh 
pip install -e .
```

Now, the package is linked to this local directory and whenever you use `import torchfem`, it will use the latest code.