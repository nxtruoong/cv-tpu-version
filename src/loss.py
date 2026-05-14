"""NT-Xent loss.

Under SPMD (single-process, multi-chip), the z @ z.t() matmul implicitly
all-gathers across the sharded batch axis via the XLA compiler — no explicit
collective needed.

`gather_distributed` flag kept for non-SPMD fallback paths (e.g. CUDA DDP)
but those paths are unused on the TPU branch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NT_XENT_TEMPERATURE


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent loss. SPMD-compatible (no explicit all_gather).

    Args:
        temperature: Softmax temperature τ.
        gather_distributed: Legacy flag for non-SPMD multi-process backends.
            Leave False on TPU SPMD — the compiler handles cross-chip gather
            inside the similarity matmul.
    """

    def __init__(
        self,
        temperature: float = NT_XENT_TEMPERATURE,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        # gather_distributed retained for API stability; ignored under SPMD.
        self.gather_distributed = gather_distributed

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: L2-normalized projections of shape (B, D)."""
        batch = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        sim = torch.matmul(z, z.t()) / self.temperature  # (2B, 2B)

        mask_self = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, float("-inf"))

        targets = torch.arange(2 * batch, device=z.device)
        targets = (targets + batch) % (2 * batch)

        loss = F.cross_entropy(sim, targets)
        return loss


def alignment_uniformity(z1: torch.Tensor, z2: torch.Tensor) -> dict:
    """Wang & Isola (2020) diagnostics. Collapse → uniformity ≈ 0."""
    with torch.no_grad():
        align = (z1 - z2).norm(dim=1).pow(2).mean().item()
        z = torch.cat([z1, z2], dim=0)
        pdist = torch.pdist(z, p=2).pow(2)
        uniform = pdist.mul(-2).exp().mean().log().item()
    return {"alignment": align, "uniformity": uniform}
