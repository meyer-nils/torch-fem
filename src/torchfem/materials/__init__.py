"""Material models.

Classes are organized into submodules but re-exported here, so
``from torchfem.materials import <Class>`` keeps working.
"""

from .base import Material
from .conductivity import (
    IsotropicConductivity1D,
    IsotropicConductivity2D,
    IsotropicConductivity3D,
    OrthotropicConductivity2D,
    OrthotropicConductivity3D,
)
from .damage import IsotropicDamage3D
from .elasticity import (
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
    OrthotropicElasticity3D,
    OrthotropicElasticityPlaneStrain,
    OrthotropicElasticityPlaneStress,
    TransverseIsotropicElasticity3D,
)
from .hyperelasticity import (
    Hyperelastic3D,
    HyperelasticPlaneStrain,
    HyperelasticPlaneStress,
)
from .plasticity import (
    IsotropicPlasticity1D,
    IsotropicPlasticity3D,
    IsotropicPlasticityPlaneStrain,
    IsotropicPlasticityPlaneStress,
)

__all__ = [
    "Material",
    "IsotropicElasticity3D",
    "IsotropicElasticityPlaneStress",
    "IsotropicElasticityPlaneStrain",
    "IsotropicElasticity1D",
    "OrthotropicElasticity3D",
    "TransverseIsotropicElasticity3D",
    "OrthotropicElasticityPlaneStress",
    "OrthotropicElasticityPlaneStrain",
    "Hyperelastic3D",
    "HyperelasticPlaneStress",
    "HyperelasticPlaneStrain",
    "IsotropicPlasticity3D",
    "IsotropicPlasticityPlaneStress",
    "IsotropicPlasticityPlaneStrain",
    "IsotropicPlasticity1D",
    "IsotropicDamage3D",
    "IsotropicConductivity3D",
    "IsotropicConductivity2D",
    "IsotropicConductivity1D",
    "OrthotropicConductivity3D",
    "OrthotropicConductivity2D",
]
