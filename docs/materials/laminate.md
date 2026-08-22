# Laminate

A `Laminate` stacks plane-stress layers into a section for a [Shell](../models/shell.md). It takes the place of the material there, and the shell integrates the layers through the thickness during the analysis rather than reducing them to ABD matrices, so a layer may be nonlinear or carry internal state:

```py
import torch
from torchfem import Laminate, Shell
from torchfem.materials import OrthotropicElasticityPlaneStress

gfrp = OrthotropicElasticityPlaneStress(
    E_1=54000.0, E_2=9400.0, nu_12=0.33, G_12=5500.0, G_13=5500.0, G_23=3000.0
)

layup = Laminate(
    materials=[gfrp] * 4,
    thicknesses=[0.25] * 4,
    angles=[0.0, torch.pi / 2, torch.pi / 2, 0.0],
)

model = Shell(nodes, elements, layup)
```

Layers are given from the bottom surface upwards. `symmetric` mirrors the half-stack about the mid-plane, and `offset` moves the reference surface the shell nodes sit on:

![Stacking sequences of three laminates](../images/laminate/laminate_stacking_light.png#only-light)
![Stacking sequences of three laminates](../images/laminate/laminate_stacking_dark.png#only-dark)

::: torchfem.Laminate
    options:
        show_root_toc_entry: false
        docstring_section_style: list
        members:
            - __init__
            - vectorize
            - plot
