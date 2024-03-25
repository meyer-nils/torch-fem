from itertools import permutations
from math import acosh

import torch

from .materials import Isotropic, Orthotropic


def IBOF_closure(A2: torch.Tensor) -> torch.Tensor:
    """IBOF closure [1]. This is a PyTorch re-implementation of fiberoripy [2].

    [1] Du Hwan Chung and Tai Hun Kwon (2002)
        'Invariant-based optimal fitting closure approximation for the numerical
        prediction of flow-induced fiber orientation',
        Journal of Rheology 46(1):169-194,
        https://doi.org/10.1122/1.1423312

    [2] Nils Meyer, Constantin Krauss & Julian Karl Bauer (2022).
        fiberoripy (v1.1.0)
        https://doi.org/10.5281/zenodo.7305587

    Args:
        A2 (torch.tensor): Second order fiber orientation tensor (shape: 3x3)

    Returns:
        torch.tensor: Fourth order fiber orientation tensor (shape: 3x3x3x3)
    """

    # second invariant
    II = (
        A2[..., 0, 0] * A2[..., 1, 1]
        + A2[..., 1, 1] * A2[..., 2, 2]
        + A2[..., 0, 0] * A2[..., 2, 2]
        - A2[..., 0, 1] * A2[..., 1, 0]
        - A2[..., 1, 2] * A2[..., 2, 1]
        - A2[..., 0, 2] * A2[..., 2, 0]
    )

    # third invariant
    III = torch.linalg.det(A2)

    # coefficients from Chung & Kwon paper
    C1 = torch.zeros((1, 21))

    C2 = torch.zeros((1, 21))

    C3 = torch.tensor(
        [
            [
                0.24940908165786e2,
                -0.435101153160329e3,
                0.372389335663877e4,
                0.703443657916476e4,
                0.823995187366106e6,
                -0.133931929894245e6,
                0.880683515327916e6,
                -0.991630690741981e7,
                -0.159392396237307e5,
                0.800970026849796e7,
                -0.237010458689252e7,
                0.379010599355267e8,
                -0.337010820273821e8,
                0.322219416256417e5,
                -0.257258805870567e9,
                0.214419090344474e7,
                -0.449275591851490e8,
                -0.213133920223355e8,
                0.157076702372204e10,
                -0.232153488525298e5,
                -0.395769398304473e10,
            ]
        ]
    )

    C4 = torch.tensor(
        [
            [
                -0.497217790110754e0,
                0.234980797511405e2,
                -0.391044251397838e3,
                0.153965820593506e3,
                0.152772950743819e6,
                -0.213755248785646e4,
                -0.400138947092812e4,
                -0.185949305922308e7,
                0.296004865275814e4,
                0.247717810054366e7,
                0.101013983339062e6,
                0.732341494213578e7,
                -0.147919027644202e8,
                -0.104092072189767e5,
                -0.635149929624336e8,
                -0.247435106210237e6,
                -0.902980378929272e7,
                0.724969796807399e7,
                0.487093452892595e9,
                0.138088690964946e5,
                -0.160162178614234e10,
            ]
        ]
    )

    C5 = torch.zeros((1, 21))

    C6 = torch.tensor(
        [
            [
                0.234146291570999e2,
                -0.412048043372534e3,
                0.319553200392089e4,
                0.573259594331015e4,
                -0.485212803064813e5,
                -0.605006113515592e5,
                -0.477173740017567e5,
                0.599066486689836e7,
                -0.110656935176569e5,
                -0.460543580680696e8,
                0.203042960322874e7,
                -0.556606156734835e8,
                0.567424911007837e9,
                0.128967058686204e5,
                -0.152752854956514e10,
                -0.499321746092534e7,
                0.132124828143333e9,
                -0.162359994620983e10,
                0.792526849882218e10,
                0.466767581292985e4,
                -0.128050778279459e11,
            ]
        ]
    )

    # build matrix of coefficients by stacking vectors
    C = torch.vstack((C1, C2, C3, C4, C5, C6))

    # compute parameters as fith order polynom based on invariants
    beta3 = (
        C[2, 0]
        + C[2, 1] * II
        + C[2, 2] * II**2
        + C[2, 3] * III
        + C[2, 4] * III**2
        + C[2, 5] * II * III
        + C[2, 6] * II**2 * III
        + C[2, 7] * II * III**2
        + C[2, 8] * II**3
        + C[2, 9] * III**3
        + C[2, 10] * II**3 * III
        + C[2, 11] * II**2 * III**2
        + C[2, 12] * II * III**3
        + C[2, 13] * II**4
        + C[2, 14] * III**4
        + C[2, 15] * II**4 * III
        + C[2, 16] * II**3 * III**2
        + C[2, 17] * II**2 * III**3
        + C[2, 18] * II * III**4
        + C[2, 19] * II**5
        + C[2, 20] * III**5
    )

    beta4 = (
        C[3, 0]
        + C[3, 1] * II
        + C[3, 2] * II**2
        + C[3, 3] * III
        + C[3, 4] * III**2
        + C[3, 5] * II * III
        + C[3, 6] * II**2 * III
        + C[3, 7] * II * III**2
        + C[3, 8] * II**3
        + C[3, 9] * III**3
        + C[3, 10] * II**3 * III
        + C[3, 11] * II**2 * III**2
        + C[3, 12] * II * III**3
        + C[3, 13] * II**4
        + C[3, 14] * III**4
        + C[3, 15] * II**4 * III
        + C[3, 16] * II**3 * III**2
        + C[3, 17] * II**2 * III**3
        + C[3, 18] * II * III**4
        + C[3, 19] * II**5
        + C[3, 20] * III**5
    )

    beta6 = (
        C[5, 0]
        + C[5, 1] * II
        + C[5, 2] * II**2
        + C[5, 3] * III
        + C[5, 4] * III**2
        + C[5, 5] * II * III
        + C[5, 6] * II**2 * III
        + C[5, 7] * II * III**2
        + C[5, 8] * II**3
        + C[5, 9] * III**3
        + C[5, 10] * II**3 * III
        + C[5, 11] * II**2 * III**2
        + C[5, 12] * II * III**3
        + C[5, 13] * II**4
        + C[5, 14] * III**4
        + C[5, 15] * II**4 * III
        + C[5, 16] * II**3 * III**2
        + C[5, 17] * II**2 * III**3
        + C[5, 18] * II * III**4
        + C[5, 19] * II**5
        + C[5, 20] * III**5
    )

    beta1 = (
        3
        / 5
        * (
            -1 / 7
            + 1 / 5 * beta3 * (1 / 7 + 4 / 7 * II + 8 / 3 * III)
            - beta4 * (1 / 5 - 8 / 15 * II - 14 / 15 * III)
            - beta6
            * (
                1 / 35
                - 24 / 105 * III
                - 4 / 35 * II
                + 16 / 15 * II * III
                + 8 / 35 * II**2
            )
        )
    )

    beta2 = (
        6
        / 7
        * (
            1
            - 1 / 5 * beta3 * (1 + 4 * II)
            + 7 / 5 * beta4 * (1 / 6 - II)
            - beta6 * (-1 / 5 + 2 / 3 * III + 4 / 5 * II - 8 / 5 * II**2)
        )
    )

    beta5 = -4 / 5 * beta3 - 7 / 5 * beta4 - 6 / 5 * beta6 * (1 - 4 / 3 * II)

    # second order identy matrix
    delta = torch.eye(3)

    # generate fourth order tensor with parameters and tensor algebra
    return (
        symm(torch.einsum("..., ij,kl->...ijkl", beta1, delta, delta))
        + symm(torch.einsum("..., ij, ...kl-> ...ijkl", beta2, delta, A2))
        + symm(torch.einsum("..., ...ij, ...kl -> ...ijkl", beta3, A2, A2))
        + symm(torch.einsum("..., ij, ...km, ...ml -> ...ijkl", beta4, delta, A2, A2))
        + symm(torch.einsum("..., ...ij, ...km, ...ml -> ...ijkl", beta5, A2, A2, A2))
        + symm(
            torch.einsum(
                "..., ...im, ...mj, ...kn, ...nl -> ...ijkl", beta6, A2, A2, A2, A2
            )
        )
    )


def symm(A4: torch.Tensor) -> torch.Tensor:
    """Symmetrize a fourth order tensor. This is a PyTorch re-implementation of
    fiberoripy [1].

    [1] Nils Meyer, Constantin Krauss & Julian Karl Bauer (2022).
        fiberoripy (v1.1.0)
        https://doi.org/10.5281/zenodo.7305587

    Args:
        A4 (torch.Tensor): Input tensor (shape Nx3x3x3x3)

    Returns:
        torch.Tensor: Symmetrized outout tensor
    """
    B4 = torch.stack([torch.permute(A4, (0, *p)) for p in permutations([1, 2, 3, 4])])
    return B4.sum(dim=0) / 24


def compute_orientation_average(
    C: torch.Tensor, A2: torch.Tensor, A4: torch.Tensor
) -> torch.Tensor:
    """Orientation averaging according to Advani and Tucker [1]. See also homopy [2].

    [1] Suresh G. Advani, Charles L. Tucker (1987)
        'The Use of Tensors to Describe and Predict Fiber Orientation in Short Fiber
        Composites',
        Journal of Rheology, 31 (8): 751-78
        https://doi.org/10.1122/1.549945

    [2] Nicolas Christ. (2023),
        homopy: v1.1.0,
        https://doi.org/10.5281/zenodo.7967631

    Args:
        C (torch.tensor): Stiffness tensor (shape 3x3x3x3)
        A2 (torch.tensor): Second order fiber orientation tensors (shape Nx3x3)
        A4 (torch.tensor): Fourth order fiber orientation tensors (shape Nx3x3x3x3)

    Returns:
        torch.tensor: Orientation averaged stiffness tensor (shape Nx3x3x3x3)
    """
    if A2.ndim == 2:
        A2 = A2.unsqueeze(0)

    # Identity tensor
    Id = torch.eye(3)

    # Coeffients from Advani-Tucker paper
    B1 = C[0, 0, 0, 0] + C[1, 1, 1, 1] - 2 * C[0, 0, 1, 1] - 4 * C[0, 1, 0, 1]
    B2 = C[0, 0, 1, 1] - C[1, 1, 2, 2]
    B3 = C[0, 1, 0, 1] + 0.5 * (C[1, 1, 2, 2] - C[1, 1, 1, 1])
    B4 = C[1, 1, 2, 2]
    B5 = 0.5 * (C[1, 1, 1, 1] - C[1, 1, 2, 2])

    # Einstein summation to fill stiffness matrix
    _C = (
        B1 * A4
        + B2
        * (
            torch.einsum("...ij,kl->...ijkl", A2, Id)
            + torch.einsum("...kl,ij->...ijkl", A2, Id)
        )
        + B3
        * (
            torch.einsum("...ik,jl->...ijkl", A2, Id)
            + torch.einsum("...il,jk->...ijkl", A2, Id)
            + torch.einsum("...jl,ik->...ijkl", A2, Id)
            + torch.einsum("...jk,il->...ijkl", A2, Id)
        )
        + B4 * torch.einsum("ij,kl->ijkl", Id, Id)
        + B5
        * (torch.einsum("ik,jl->ijkl", Id, Id) + torch.einsum("il,jk->ijkl", Id, Id))
    )

    C = torch.zeros(_C.shape[0], 6, 6)
    C[:, 0, 0] = _C[:, 0, 0, 0, 0]
    C[:, 0, 1] = _C[:, 0, 0, 1, 1]
    C[:, 0, 2] = _C[:, 0, 0, 2, 2]
    C[:, 1, 0] = _C[:, 1, 1, 0, 0]
    C[:, 1, 1] = _C[:, 1, 1, 1, 1]
    C[:, 1, 2] = _C[:, 1, 1, 2, 2]
    C[:, 2, 0] = _C[:, 2, 2, 0, 0]
    C[:, 2, 1] = _C[:, 2, 2, 1, 1]
    C[:, 2, 2] = _C[:, 2, 2, 2, 2]
    C[:, 3, 3] = _C[:, 1, 2, 1, 2]
    C[:, 4, 4] = _C[:, 0, 2, 0, 2]
    C[:, 5, 5] = _C[:, 0, 1, 0, 1]
    return C


def tandon_weng_homogenization(
    matrix: Isotropic, fiber: Isotropic, a, volfrac
) -> Orthotropic:
    """Compute transversly isotropic material based on Tandon-Wengs's paper [1]. See
    also [2] for more general Mori-Tanka approach.

    [1] Tandon, G.P. and Weng, G.J. (1984)
        'The effect of aspect ratio of inclusions on the elastic properties of
        unidirectionally aligned composites',
        Polymer Composites, 5: 327-333,
        https://doi.org/10.1002/pc.750050413

    [2] Nicolas Christ. (2023),
        homopy: v1.1.0,
        https://doi.org/10.5281/zenodo.7967631


    Args:
        matrix (Isotropic): Isotropic matrix material
        fiber (Isotropic): Isotropic fiber material
        a (float): Aspect ratio of inclusion
        volfrac (float): Volume fraction of inclusions

    Returns:
        Otrhotropic: Transversely isotropic effective elastic material.
    """

    # Extract scalar properties from matrix and fiber material
    lambda0 = matrix.lbd()
    G0 = matrix.G()
    E0 = matrix.E()
    nu0 = matrix.nu()
    lambda1 = fiber.lbd()
    G1 = fiber.G()

    # Utility variables
    b = a**2 - 1
    c = 1 - nu0
    g = a / (b**1.5) * (a * b ** (1 / 2) - acosh(a))

    # Eshelby tensor
    S = torch.zeros((3, 3, 3, 3))
    S[0, 0, 0, 0] = (
        1
        / (2 * c)
        * (1 - 2 * nu0 + (3 * a**2 - 1) / b - (1 - 2 * nu0 + (3 * a**2) / b) * g)
    )
    S[1, 1, 1, 1] = (
        3 / (8 * c) * a**2 / b + 1 / (4 * c) * (1 - 2 * nu0 - 9 / (4 * b)) * g
    )
    S[1, 1, 2, 2] = 1 / (4 * c) * (a**2 / (2 * b) - (1 - 2 * nu0 + 3 / (4 * b)) * g)
    S[1, 1, 0, 0] = (
        -1 / (2 * c) * a**2 / b + 1 / (4 * c) * (3 * a**2 / b - (1 - 2 * nu0)) * g
    )
    S[0, 0, 1, 1] = (
        -1 / (2 * c) * (1 - 2 * nu0 + 1 / b)
        + 1 / (2 * c) * (1 - 2 * nu0 + 3 / (2 * b)) * g
    )
    S[1, 2, 1, 2] = 1 / (4 * c) * (a**2 / (2 * b) + (1 - 2 * nu0 - 3 / (4 * b)) * g)
    S[0, 1, 0, 1] = (
        1
        / (4 * c)
        * (1 - 2 * nu0 - (a**2 + 1) / b - 0.5 * (1 - 2 * nu0 - 3 * (a**2 + 1) / b) * g)
    )
    S[2, 2, 2, 2] = S[1, 1, 1, 1]
    S[2, 2, 1, 1] = S[1, 1, 2, 2]
    S[0, 0, 2, 2] = S[0, 0, 1, 1]
    S[2, 1, 2, 1] = S[1, 2, 1, 2]
    S[2, 2, 0, 0] = S[1, 1, 0, 0]
    S[0, 2, 0, 2] = S[0, 1, 0, 1]

    # D Coefficients
    D1 = 1 + 2 * (G1 - G0) / (lambda1 - lambda0)
    D2 = (lambda0 + 2 * G0) / (lambda1 - lambda0)
    D3 = lambda0 / (lambda1 - lambda0)

    # B Coefficients
    B1 = volfrac * D1 + D2 + (1 - volfrac) * (D1 * S[0, 0, 0, 0] + 2 * S[1, 1, 0, 0])
    B2 = (
        volfrac
        + D3
        + (1 - volfrac) * (D1 * S[0, 0, 1, 1] + S[1, 1, 1, 1] + S[1, 1, 2, 2])
    )
    B3 = volfrac + D3 + (1 - volfrac) * (S[0, 0, 0, 0] + (1 + D1) * S[1, 1, 0, 0])
    B4 = (
        volfrac * D1
        + D2
        + (1 - volfrac) * (S[0, 0, 1, 1] + D1 * S[1, 1, 1, 1] + S[1, 1, 2, 2])
    )
    B5 = (
        volfrac
        + D3
        + (1 - volfrac) * (S[0, 0, 1, 1] + S[1, 1, 1, 1] + D1 * S[1, 1, 2, 2])
    )

    # A Coefficients
    A1 = D1 * (B4 + B5) - 2 * B2
    A2 = (1 + D1) * B2 - (B4 + B5)
    A3 = B1 - D1 * B3
    A4 = (1 + D1) * B1 - 2 * B3
    A5 = (1 - D1) / (B4 - B5)
    A = 2 * B2 * B3 - B1 * (B4 + B5)

    # Engineering constants
    E11 = E0 / (1 + volfrac * (A1 + 2 * nu0 * A2) / A)
    E22 = E0 / (1 + volfrac * (-2 * nu0 * A3 + c * A4 + (1 + nu0) * A5 * A) / (2 * A))
    E33 = E22
    G12 = G0 * (1 + volfrac / (G0 / (G1 - G0) + 2 * (1 - volfrac) * S[0, 1, 0, 1]))
    G23 = G0 * (1 + volfrac / (G0 / (G1 - G0) + 2 * (1 - volfrac) * S[1, 2, 1, 2]))
    G13 = G12
    nu12 = (nu0 * A - volfrac * (A3 - nu0 * A4)) / (A + volfrac * (A1 + 2 * nu0 * A2))
    nu23 = E22 / (2 * G23) - 1
    nu13 = nu12

    return Orthotropic(E11, E22, E33, nu12, nu13, nu23, G12, G13, G23)
