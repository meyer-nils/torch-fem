import torch


def box_constrained_decent(
    func, x_init, x_lower, x_upper, eta=0.1, max_iter=100, tol=1e-10
):
    """Box-constrained gradient decent algorithm.

    Args:
        func (callable): Function to be minimized.
        x_init (torch.tensor): Initial variable.
        x_lower (torch.tensor): Lower bounds on x.
        x_upper (torcht.tensor): Upper bound on x.
        eta (float, optional): Step width. Defaults to 0.1.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-10.

    Returns:
        torch.tensor: Optimal x within bounds x_lower, x_upper.
    """
    x = x_init.clone().requires_grad_()
    for i in range(max_iter):
        x_old = x.clone()
        grad = torch.autograd.grad(func(x).sum(), x)[0]
        x = x - eta * grad
        x = torch.max(torch.min(x, x_upper), x_lower)
        if torch.norm(x - x_old) < tol:
            print(f"Box constrained decent reached tolerance {tol} after {i} steps.")
            return x
    print(f"Box constrained decent reached maximum iteration count {max_iter}.")
    return x


def MMA(func, x_k, L_k, U_k):
    """Convex approximation with method of moving asymptotes (MMA)

    Args:
        func (callable): The function to be approximated.
        x_k (torch.tensor): Approximation point.
        L_k (torch.tensor): Lower asymptotes.
        U_k (torch.tensor): Upper asymptotes.

    Returns:
        callable: The approximation function.
    """
    x_lin = x_k.clone().requires_grad_()
    grads = torch.autograd.grad(func(x_lin), x_lin)[0]
    f_k = func(x_k)

    def approximation(x):
        res = f_k * torch.ones_like(x[..., 0])
        for j, grad in enumerate(grads):
            if grad < 0.0:
                p = 0
                q = -((x_k[j] - L_k[j]) ** 2) * grad
            else:
                p = (U_k[j] - x_k[j]) ** 2 * grad
                q = 0
            res -= p / (U_k[j] - x_k[j]) + q / (x_k[j] - L_k[j])
            res += p / (U_k[j] - x[..., j]) + q / (x[..., j] - L_k[j])
        return res

    return approximation
