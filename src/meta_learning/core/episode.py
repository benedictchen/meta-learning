from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class Episode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor

    def validate(self, *, expect_n_classes: Optional[int] = None) -> None:
        assert self.support_x.shape[0] == self.support_y.shape[0], "support X/Y mismatch"
        assert self.query_x.shape[0] == self.query_y.shape[0], "query X/Y mismatch"
        assert self.support_y.dtype == torch.int64 and self.query_y.dtype == torch.int64, "labels must be int64"
        assert self.support_y.dim() == 1 and self.query_y.dim() == 1, "labels must be 1D"
        classes = torch.unique(self.support_y)
        assert torch.all(torch.isin(torch.unique(self.query_y), classes)), "query labels not subset of support"
        if expect_n_classes is not None:
            assert classes.numel() == expect_n_classes, f"expected {expect_n_classes} classes, got {classes.numel()}"
        remapped = torch.sort(classes).values
        assert torch.equal(remapped, torch.arange(remapped.numel(), device=remapped.device)), "labels must be [0..C-1] contiguous"

def remap_labels(y_support: torch.Tensor, y_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(y_support)
    mapping = {c.item(): i for i, c in enumerate(classes)}
    ys = torch.tensor([mapping[int(c.item())] for c in y_support], device=y_support.device)
    yq = torch.tensor([mapping[int(c.item())] for c in y_query], device=y_query.device)
    return ys.long(), yq.long()
