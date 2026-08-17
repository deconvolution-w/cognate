from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD_IDX


class BlosumLoss(nn.Module):
    def __init__(self, matrix, aa_token_ids, temperature: float) -> None:
        super().__init__()
        self.matrix = matrix
        self.temperature = float(temperature)
        aa_ids = torch.tensor(aa_token_ids, dtype=torch.long)
        aa_mask = torch.zeros(matrix.size(0), dtype=torch.bool)
        aa_mask[aa_ids] = True
        self.register_buffer("aa_ids", aa_ids, persistent=False)
        self.register_buffer("aa_mask", aa_mask, persistent=False)

    def forward(self, logits, targets):
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        valid = (targets != PAD_IDX) & self.aa_mask[targets]
        if not valid.any():
            return logits.sum() * 0.0

        soft_targets = F.softmax(
            self.matrix[targets[valid]][:, self.aa_ids] / self.temperature, dim=-1
        )
        log_probs = F.log_softmax(logits[valid][:, self.aa_ids], dim=-1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")


class PhysicochemicalLoss(nn.Module):
    def __init__(self, matrix, aa_token_ids, variance_weight: float) -> None:
        super().__init__()
        self.matrix = matrix
        self.variance_weight = float(variance_weight)
        aa_mask = torch.zeros(matrix.size(0), dtype=torch.bool)
        aa_mask[torch.tensor(aa_token_ids, dtype=torch.long)] = True
        self.register_buffer("aa_mask", aa_mask, persistent=False)

    @staticmethod
    def _mean_var(values, mask):
        mask = mask.unsqueeze(-1).float()
        count = mask.sum(dim=1).clamp(min=1.0)
        mean = (values * mask).sum(dim=1) / count
        variance = ((values * values) * mask).sum(dim=1) / count - mean * mean
        return mean, variance

    def forward(self, logits, targets):
        valid = (targets != PAD_IDX) & self.aa_mask[targets]
        if not valid.any():
            return logits.sum() * 0.0

        predicted = F.softmax(logits, dim=-1) @ self.matrix
        expected = self.matrix[targets]
        pred_mean, pred_var = self._mean_var(predicted, valid)
        true_mean, true_var = self._mean_var(expected, valid)
        return F.mse_loss(pred_mean, true_mean) + self.variance_weight * F.mse_loss(
            pred_var, true_var
        )


class ReciprocalLoss(nn.Module):
    def __init__(self, config, blosum_matrix, property_matrix, aa_token_ids) -> None:
        super().__init__()
        blosum = config.get("blosum", {})
        physicochemical = config.get("physicochemical", {})

        self.beta_kl = 0.0
        self.blosum_weight = float(blosum.get("weight_start", 0.05))
        self.blosum_start = self.blosum_weight
        self.blosum_end = float(blosum.get("weight_end", 0.0))
        self.blosum_decay_start = float(blosum.get("decay_start_frac", 0.3))
        self.blosum_decay_end = float(blosum.get("decay_end_frac", 0.8))
        self.physicochemical_weight = float(physicochemical.get("weight", 0.1))

        self.reconstruction = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.blosum = BlosumLoss(
            blosum_matrix,
            aa_token_ids,
            temperature=float(blosum.get("temperature", 2.0)),
        )
        self.physicochemical = PhysicochemicalLoss(
            property_matrix,
            aa_token_ids,
            variance_weight=float(physicochemical.get("variance_weight", 0.5)),
        )

    def update_step(
        self,
        step: int,
        total_steps: int,
        beta_max: float,
        kl_anneal_steps: int,
    ) -> None:
        self.beta_kl = float(beta_max) * min(1.0, step / max(1, kl_anneal_steps))
        progress = step / max(1, total_steps)
        if progress <= self.blosum_decay_start:
            self.blosum_weight = self.blosum_start
        elif progress >= self.blosum_decay_end:
            self.blosum_weight = self.blosum_end
        else:
            fraction = (progress - self.blosum_decay_start) / (
                self.blosum_decay_end - self.blosum_decay_start
            )
            self.blosum_weight = self.blosum_start + fraction * (
                self.blosum_end - self.blosum_start
            )

    @staticmethod
    def _kl(mu_post, logvar_post, mu_prior, logvar_prior):
        value = logvar_prior - logvar_post
        value += (
            torch.exp(logvar_post) + (mu_post - mu_prior).pow(2)
        ) / torch.exp(logvar_prior)
        return (0.5 * (value - 1)).sum(dim=-1).mean()

    def forward(self, model_output, target):
        logits = model_output["logits"]
        length = min(logits.size(1), target.size(1))
        logits = logits[:, :length]
        target = target[:, :length]

        reconstruction = self.reconstruction(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )
        kl = self._kl(
            model_output["mu_post"],
            model_output["logvar_post"],
            model_output["mu_prior"],
            model_output["logvar_prior"],
        )
        blosum = self.blosum(logits, target)
        physicochemical = self.physicochemical(logits, target)
        total = (
            reconstruction
            + self.beta_kl * kl
            + self.blosum_weight * blosum
            + self.physicochemical_weight * physicochemical
        )
        return {
            "loss": total,
            "reconstruction": reconstruction.detach(),
            "kl": kl.detach(),
            "blosum": blosum.detach(),
            "physicochemical": physicochemical.detach(),
        }
