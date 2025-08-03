import torch
from torch import Tensor
from typing import Optional, Tuple

def nbs_ccp(
    gram: Tensor,
    alpha0: Optional[Tensor] = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> Tuple[Tensor, int]:
    """
    Solve M·α = 1/α  (component‑wise reciprocal) via a log‑barrier Newton/CCP scheme.
    
    Parameters
    ----------
    gram : Tensor[K, K]
        Symmetric positive‑semidefinite Gram matrix GᵀG of task gradients.
        Must reside on the desired device (CPU or CUDA).
    alpha0 : Tensor[K], optional
        Positive initial guess.  If None, uses ones_like(gram[..., 0]).
    max_iter : int
        Newton iterations (20 ≈ paper; often ≤10 is enough).
    tol : float
        Infinity‑norm convergence tolerance ‖Mα − 1/α‖∞.
    eps : float
        Minimal α to keep strict positivity (prevents division by zero).

    Returns
    -------
    alpha : Tensor[K]
        Positive Nash weight vector on the same device/dtype as `gram`.
    n_iter : int
        Number of Newton steps actually performed.
    """
    K = gram.size(0)
    if alpha0 is None:
        alpha = torch.ones(K, device=gram.device, dtype=gram.dtype)
    else:
        alpha = alpha0.clone().to(gram.device, gram.dtype)

    for it in range(max_iter):
        # Gradient  g = 2Mα − 2/α
        inv_alpha = 1.0 / alpha
        grad = 2.0 * (gram @ alpha) - 2.0 * inv_alpha

        # Stopping test  ‖Mα − 1/α‖∞
        if torch.max(torch.abs(grad * 0.5)) < tol:
            return alpha, it

        # Hessian  H = 2M + 2 diag(1/α²)
        hess = 2.0 * gram + 2.0 * torch.diag(inv_alpha * inv_alpha)

        # Newton step:   H δ = g
        delta = torch.linalg.solve(hess, grad)

        # Damped update to ensure positivity
        step = 1.0
        while step > 1e-4:
            new_alpha = torch.clamp(alpha - step * delta, min=eps)
            if torch.isfinite(new_alpha).all():
                alpha = new_alpha
                break
            step *= 0.5  # back‑tracking if numerical problems

    return alpha, max_iter
