---
icon: lucide/rocket
---

# Installation
Your may install torch-fem via pip with
```
pip install torch-fem
```

*Optional*: For GPU support, install CUDA, PyTorch for CUDA, and the corresponding CuPy version.

For CUDA 11.8: 
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x # v11.2 - 11.8
```

For CUDA 12.6:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install cupy-cuda12x # v12.x
```