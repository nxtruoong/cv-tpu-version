"""NT-Xent loss.

On TPU SPMD with batch-sharded inputs, each chip only holds a local shard of
`z1`/`z2`. Relying on the XLA compiler to infer a global `z @ z.t()` from
sharded activations is fragile across PJRT / v5e stacks and can yield wrong
contrastive logits (collapse, flat diagnostics, or layout-related corruption).

When ``gather_distributed=True``, we explicitly ``xm.all_gather`` on dim 0 so
NT-Xent always matches the global batch (same semantics as multi-GPU SimCLR).
Set env ``XLA_ALL_GATHER_PIN_LAYOUT=0`` if you hit XLA layout compile errors
(pin_layout can trade compile fragility vs. stricter layout matching).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NT_XENT_TEMPERATURE


def _xla_all_gather_cat_dim0(t: torch.Tensor) -> torch.Tensor:
    if t.device.type != "xla":
        return t
    import torch_xla.core.xla_model as xm

    pin_layout = os.environ.get("XLA_ALL_GATHER_PIN_LAYOUT", "1") != "0"
    return xm.all_gather(t, dim=0, pin_layout=pin_layout)


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent loss.

    Args:
        temperature: Softmax temperature τ.
        gather_distributed: If True (recommended for TPU SPMD data-parallel),
            all-gather normalized projections on the batch axis before building
            the (2B, 2B) similarity matrix. If False, loss uses only the local
            shard (wrong global batch size under input sharding).
    """

    def __init__(
        self,
        temperature: float = NT_XENT_TEMPERATURE,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: L2-normalized projections of shape (B_local, D) or (B, D)."""
        if self.gather_distributed:
            z1 = _xla_all_gather_cat_dim0(z1)
            z2 = _xla_all_gather_cat_dim0(z2)

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
