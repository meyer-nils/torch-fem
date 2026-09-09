"""Material models.

Classes are organized into submodules but re-exported here, so
``from torchfem.materials import <Class>`` keeps working.
"""

from .base import HeatMaterial, Material, MechanicsMaterial
from .conductivity import (
    IsotropicConductivity2D,
    IsotropicConductivity3D,
    OrthotropicConductivity2D,
    OrthotropicConductivity3D,
)
from .damage import (
    IsotropicDamage3D,
    IsotropicDamagePlaneStrain,
    IsotropicDamagePlaneStress,
)
from .elasticity import (
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
    OrthotropicElasticity3D,
    OrthotropicElasticityPlaneStrain,
    OrthotropicElasticityPlaneStress,
    TransverseIsotropicElasticity3D,
    TransverseIsotropicElasticityPlaneStrain,
    TransverseIsotropicElasticityPlaneStress,
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
    "MechanicsMaterial",
    "HeatMaterial",
    "IsotropicElasticity3D",
    "IsotropicElasticityPlaneStress",
    "IsotropicElasticityPlaneStrain",
    "IsotropicElasticity1D",
    "OrthotropicElasticity3D",
    "TransverseIsotropicElasticity3D",
    "TransverseIsotropicElasticityPlaneStress",
    "TransverseIsotropicElasticityPlaneStrain",
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
    "IsotropicDamagePlaneStrain",
    "IsotropicDamagePlaneStress",
    "IsotropicConductivity3D",
    "IsotropicConductivity2D",
    "OrthotropicConductivity3D",
    "OrthotropicConductivity2D",
]
