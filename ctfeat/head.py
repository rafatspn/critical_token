"""Prediction-head geometry: W_U singular basis, centered Φ-Align basis, final RMSNorm."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PredictionHead:
    weight: torch.Tensor       # W_U, vocab x d, float32
    norm_weight: torch.Tensor  # g, d
    eps: float
    basis: torch.Tensor        # right singular vectors of W_U (d x d), by singular value, descending
    phi_basis: torch.Tensor    # eigenvectors v_j of W_cᵀW_c (d x d), descending
    phi_strengths: torch.Tensor  # s_j² (d,)

    @property
    def d(self) -> int:
        return self.weight.shape[1]

    @property
    def phi_frobenius(self) -> float:
        """‖W_c‖²_F = Σ_j s_j²."""
        return float(self.phi_strengths.sum())

    @classmethod
    def from_weights(cls, weight: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> PredictionHead:
        weight = weight.float()
        gram = weight.T @ weight
        values, vectors = torch.linalg.eigh(gram.double())
        basis = vectors.flip(-1).float()
        mean_row = weight.mean(0, dtype=torch.float64)
        # W_cᵀW_c = W_UᵀW_U − |V| w̄w̄ᵀ: removes the softmax-invariant shared logit shift.
        centered = gram.double() - weight.shape[0] * torch.outer(mean_row, mean_row)
        strengths, phi_vectors = torch.linalg.eigh(centered)
        return cls(
            weight=weight, norm_weight=norm_weight.float(), eps=float(eps), basis=basis,
            phi_basis=phi_vectors.flip(-1).float(), phi_strengths=strengths.flip(-1).clamp_min(0).float(),
        )

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> PredictionHead:
        norm = model.model.norm
        return cls.from_weights(model.lm_head.weight.detach(), norm.weight.detach(),
                                getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))

    def rms(self, h: torch.Tensor) -> torch.Tensor:
        """The final RMSNorm's scale s = sqrt(mean(h²) + eps), per row."""
        h = h.float()
        return torch.sqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)

    def normed(self, h: torch.Tensor) -> torch.Tensor:
        """ĥ: final RMSNorm including its weight, so logit(y) = ĥ · w_y."""
        return h.float() / self.rms(h) * self.norm_weight

    def logits(self, h_hat: torch.Tensor) -> torch.Tensor:
        return h_hat @ self.weight.T

    def readout_frac(self, h_hat: torch.Tensor, k: int) -> torch.Tensor:
        coords = h_hat @ self.basis[:, :k]
        return coords.pow(2).sum(-1) / h_hat.pow(2).sum(-1)

    def cand_span_resid(self, h_hat: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """1 − ‖proj(ĥ onto span{w_c})‖² / ‖ĥ‖² for each row's candidate tokens (rows x k)."""
        q, _ = torch.linalg.qr(self.weight[candidates].transpose(1, 2))  # rows x d x k
        projected = torch.einsum("rdk,rd->rk", q, h_hat)
        return 1 - projected.pow(2).sum(-1) / h_hat.pow(2).sum(-1)

    def phi_coords(self, x: torch.Tensor) -> torch.Tensor:
        """u = Vᵀx in the centered head's eigenbasis."""
        return x.float() @ self.phi_basis

    def phi_tok(self, x: torch.Tensor) -> torch.Tensor:
        """‖W_c x‖² / (‖x‖² ‖W_c‖²_F) per row."""
        u = self.phi_coords(x)
        return (self.phi_strengths * u.pow(2)).sum(-1) / (u.pow(2).sum(-1) * self.phi_frobenius)
