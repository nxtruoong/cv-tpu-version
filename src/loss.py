"""NT-Xent loss with XLA-aware all_gather.

XLA path: xm.all_gather preserves gradients natively across replicas — no need
to splice the local slot back in (unlike NCCL where all_gather is detached).

For single-device use, gather is a no-op and this reduces to standard NT-Xent.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NT_XENT_TEMPERATURE


def _xla_world_size() -> int:
    try:
        import torch_xla.runtime as xr
        return xr.world_size()
    except Exception:
        return 1


def _all_gather_xla(t: torch.Tensor) -> torch.Tensor:
    """XLA all_gather; preserves grad for local rank."""
    import torch_xla.core.xla_model as xm
    return xm.all_gather(t, dim=0)


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent loss with XLA multi-core support.

    Args:
        temperature: Softmax temperature τ.
        gather_distributed: If True, gather z from all replicas before computing
            loss so anchors see negatives from all cores (true batch).
    """

    def __init__(
        self,
        temperature: float = NT_XENT_TEMPERATURE,
        gather_distributed: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: L2-normalized projections of shape (B, D)."""
        if self.gather_distributed and _xla_world_size() > 1:
            z1 = _all_gather_xla(z1)
            z2 = _all_gather_xla(z2)

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
