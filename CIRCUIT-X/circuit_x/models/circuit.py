"""Binary-mask sparse subnetwork wrapper."""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CircuitMask(nn.Module):
    """Differentiable sigmoid-gated mask over a set of candidate parameter groups.

    Each group gets one scalar logit; soft gate = sigmoid(logit),
    hard gate = 1 if sigmoid > 0.5 else 0.
    """

    def __init__(self, candidate_groups: List[Dict]):
        super().__init__()
        self.groups = candidate_groups
        # Initialise mask logits near 1 → sigmoid(3) ≈ 0.95 (start mostly open)
        self.mask_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(3.0)) for _ in candidate_groups
        ])

    def soft_gates(self) -> List[torch.Tensor]:
        return [torch.sigmoid(m) for m in self.mask_logits]

    def hard_gates(self) -> List[float]:
        return [1.0 if torch.sigmoid(m).item() > 0.5 else 0.0
                for m in self.mask_logits]

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of soft gates."""
        gates = torch.stack(self.soft_gates())
        return gates.sum()

    def apply_hard_mask(self, model: nn.Module):
        """Zero out parameters whose hard gate = 0 (in-place, permanent)."""
        gates = self.hard_gates()
        for gate, group in zip(gates, self.groups):
            if gate == 0.0:
                for _name, param in group["params"]:
                    param.data.zero_()
        active = sum(g == 1.0 for g in gates)
        logger.info(f"Hard mask applied: {active}/{len(gates)} groups active.")

    def parameter_efficiency(self, total_params: int) -> float:
        """Fraction of active parameters over total parameters."""
        gates = self.hard_gates()
        active_params = sum(
            sum(p.numel() for _, p in grp["params"])
            for gate, grp in zip(gates, self.groups)
            if gate == 1.0
        )
        return active_params / max(total_params, 1)

    def active_groups(self) -> List[Dict]:
        """Return only the groups whose hard gate = 1."""
        return [grp for gate, grp in zip(self.hard_gates(), self.groups)
                if gate == 1.0]


class MaskedModel(nn.Module):

    def __init__(self, base_model: nn.Module, circuit_mask: CircuitMask):
        super().__init__()
        self.base_model   = base_model
        self.circuit_mask = circuit_mask
        self._hooks: list = []

        for param in self.base_model.parameters():
            param.requires_grad_(False)

    def _register_hooks(self, soft_gates: List[torch.Tensor]):
        """Register forward hooks that scale module outputs by soft gate."""
        self._remove_hooks()
        for gate, group in zip(soft_gates, self.circuit_mask.groups):
            module = self._get_module(group["name"])
            if module is None:
                continue

            def make_hook(g):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        return output * g
                    if isinstance(output, tuple):
                        return (output[0] * g,) + output[1:]
                    return output
                return hook

            h = module.register_forward_hook(make_hook(gate))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _get_module(self, name: str):
        parts = name.split(".")
        mod = self.base_model
        for part in parts:
            mod = getattr(mod, part, None)
            if mod is None:
                return None
        return mod

    def forward(self, **kwargs):
        soft_gates = self.circuit_mask.soft_gates()
        self._register_hooks(soft_gates)
        try:
            out = self.base_model(**kwargs)
        finally:
            self._remove_hooks()
        return out

    def __del__(self):
        self._remove_hooks()
