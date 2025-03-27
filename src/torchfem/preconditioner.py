import torch
from torch import Tensor

try:
    import cupy
    from cupyx.scipy.sparse import coo_matrix as cupy_coo_matrix
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
    from cupyx.scipy.sparse import eye as cupy_eye
    from cupyx.scipy.sparse.linalg import spsolve_triangular

    cupy_available = False
except ImportError:
    cupy_available = False

import numpy as np


def ssor_preconditioner(A: Tensor, omega: int=1, filter: float=1.e-9, full_solve=False) -> Tensor:
    """
        Calculate the SSOR Preconditioner from https://www.netlib.org/linalg/html_templates/node58.html#SECTION00830000000000000000
        to see if it reduces condition number below Jacobi

        Input:
            A: Tensor (N x N): The Sparse Linear System we would like to solve over
            omega: int: The relaxation factor
    """
    # It shouldn't need to do this, but having gradient information will cause memory explosion
    if(A.requires_grad):
        A = A.detach()

    # Do numpy implementation just for checking (not actually useful)
    if(not A.is_cuda and not cupy_available):
        D = np.diag(np.diag(A))/omega
        D_inv = np.diag(omega/np.diag(A))
        L = np.tril(A,k=-1)

        # Unfortunately need this explicityly which means we can't do this on large systems.
        D_L_inv = np.linalg.inv(D + L)

        M = 1./(2 - omega) * np.linalg.matmul(np.linalg.matmul(D_L_inv, D), np.transpose(D_L_inv))
        return M

    else:
        # Get indices and values
        A = A.coalesce()

        # Get indices and values from A
        idxs = A.indices()
        values = A.values()

        # Diagonal is where i == j
        diag_mask = idxs[0] == idxs[1]
        D = torch.sparse_coo_tensor(idxs[:,diag_mask], (1./omega) * values[diag_mask], A.shape)

        if(full_solve):
            # Put A in CuPy sparse form
            A = cupy_coo_matrix(
                    (
                        cupy.asarray(A._values()),
                        (cupy.asarray(A._indices()[0]), cupy.asarray(A._indices()[1])),
                    ),
            )

            ###
            # WARNING:
            # Construct dense identity matrix because that's what spsolve_triangular needs
            # This is the bottleneck. It's currently unavoidable with what CuPy supports.
            ###
            other = cupy.eye(A.shape[0])
            D_L_inv = spsolve_triangular(A, other)
            D_L_inv.eliminate_zeros()       

        else: # Using Neumanan expansion to calculate approximate inverse of (D + L)

            # Explicitly construct L here, where lower diagonal is when i > j
            l_mask = idxs[0] > idxs[1] 
            L = torch.sparse_coo_tensor(idxs[:,l_mask], values[l_mask], A.shape)

            # Put L and D_inv in cupy CSR format
            L = cupy_csr_matrix(
                    (
                        cupy.asarray(L._values()),
                        (cupy.asarray(L._indices()[0]), cupy.asarray(L._indices()[1])),
                    ),
                    shape=A.shape
            )

            # Explicitly Construct D_inv
            D_inv = torch.sparse_coo_tensor(idxs[:,diag_mask], omega/values[diag_mask], A.shape)
            D_inv = cupy_csr_matrix(
                    (
                        cupy.asarray(D_inv._values()),
                        (cupy.asarray(D_inv._indices()[0]), cupy.asarray(D_inv._indices()[1])),
                    ),
                    shape=A.shape
            )

            # Use Neumann series to compute inverse
            LD_prod = L @ D_inv
            D_L_inv = cupy_eye(A.shape[0], format='csr')
            term = cupy_eye(A.shape[0], format='csr')

            ###
            # Setting the number of iterations here is critical to being faster than Jacobi preconditioner
            # Issue: Repeated multiplication leads to a lot of small values at new indices
            # Solution: Filter them out and only do one iteration which still works
            ###
            for i in range(1, 2): #TODO Can tune this to be higher.
            #for i in range(1, 3): #TODO Can tune this to be higher.
                term = -term @ LD_prod
                D_L_inv += term

                # Filter term values (Let's see if we can do this sticking only to CSR)
                term = term.tocoo() # Let's try to bypass this...
                term.sum_duplicates()
                term.eliminate_zeros()
                term_values = term.data
                term_mask = cupy.abs(term_values) > filter
                term_indices = term.row[term_mask], term.col[term_mask]
                term = cupy_csr_matrix((term_values[term_mask], term_indices), shape=A.shape)

                # Filter D_L_inv values
                D_L_inv = D_L_inv.tocoo()
                D_L_inv.sum_duplicates()
                D_L_inv.eliminate_zeros()
                D_L_inv_values = D_L_inv.data
                D_L_inv_mask = cupy.abs(D_L_inv_values) > filter
                D_L_inv_indices = D_L_inv.row[D_L_inv_mask], D_L_inv.col[D_L_inv_mask]
                D_L_inv = cupy_csr_matrix((D_L_inv_values[D_L_inv_mask], D_L_inv_indices), shape=A.shape)

            # Get final inverse approximation
            D_L_inv = D_inv @ D_L_inv

            # Filter small elements out of final D_L_inv
            D_L_inv = D_L_inv.tocoo()
            D_L_inv.sum_duplicates()
            D_L_inv.eliminate_zeros()
            D_L_inv_values = D_L_inv.data
            D_L_inv_mask = cupy.abs(D_L_inv_values) > (filter ** 1.e2)
            D_L_inv_indices = D_L_inv.row[D_L_inv_mask], D_L_inv.col[D_L_inv_mask]

            D_L_inv = cupy_csr_matrix((D_L_inv_values[D_L_inv_mask], D_L_inv_indices), shape=A.shape)
            D_L_inv.eliminate_zeros()

        # Put into CUPY sparse format
        D = cupy_coo_matrix(
                (
                    cupy.asarray(D._values()),
                    (cupy.asarray(D._indices()[0]), cupy.asarray(D._indices()[1])),
                )
        )

        # This seems to be esssentially the same.
        left = (2. - omega) * (D_L_inv).T
        right = D @ (D_L_inv) 

        # This line often times leads to memory errors.
        # A lower triangular and upper triangular matrix are being multiplied here
        # greatly increasing the number of nonzero entries.
        M = left @ right

        torch.cuda.empty_cache()

        return M
