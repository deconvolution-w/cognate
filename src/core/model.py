from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import EsmModel

from .lora import apply_lora_to_model
from .modules import CognateDecoder, CognateEncoder


class CognateCVAE(nn.Module):
    def __init__(self, vocab_size: int, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        d_model = int(model_config["d_model"])
        nhead = int(model_config["nhead"])
        num_layers = int(model_config["num_layers"])
        dropout = float(model_config["dropout"])
        latent_dim = int(model_config["latent_dim"])

        esm_config = model_config.get("esm", {})
        self.esm_enabled = bool(esm_config.get("enabled", False))
        self.esm_model = None
        if self.esm_enabled:
            self.esm_model = EsmModel.from_pretrained(
                esm_config["model_name"],
                add_pooling_layer=False,
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
            )
            self.esm_model.requires_grad_(False)
            if esm_config.get("gradient_checkpointing", False):
                self.esm_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                self.esm_model.enable_input_require_grads()
            if esm_config.get("lora", {}).get("enabled", True):
                apply_lora_to_model(self.esm_model, esm_config["lora"])

        esm_trainable = self.esm_model is not None and any(
            parameter.requires_grad for parameter in self.esm_model.parameters()
        )
        self.condition_encoder = CognateEncoder(
            vocab_size,
            d_model,
            nhead,
            num_layers,
            dropout,
            esm_model=self.esm_model,
            esm_trainable=esm_trainable,
            esm_scalar_mix_k=int(esm_config.get("scalar_mix_layers", 4)),
        )
        self.target_encoder = CognateEncoder(
            vocab_size,
            d_model,
            nhead,
            num_layers=int(model_config["target_encoder"]["num_layers"]),
            dropout=dropout,
            use_segment_embedding=False,
        )
        self.prior_net_mu = nn.Linear(d_model, latent_dim)
        self.prior_net_logvar = nn.Linear(d_model, latent_dim)
        self.post_net_mu = nn.Linear(d_model * 2, latent_dim)
        self.post_net_logvar = nn.Linear(d_model * 2, latent_dim)
        self.z_projector = nn.Linear(latent_dim, d_model)
        self.decoder = CognateDecoder(
            vocab_size, d_model, nhead, num_layers, dropout
        )

    @staticmethod
    def _pool(sequence, padding_mask):
        if padding_mask is None:
            return sequence.mean(dim=1)
        valid = (~padding_mask).unsqueeze(-1).float()
        return (sequence * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def reparameterize(mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_pad_mask,
        tgt_causal_mask=None,
        segment_ids=None,
        esm_hla_src=None,
        esm_hla_mask=None,
        esm_pep_src=None,
        esm_pep_mask=None,
        hla_len=None,
        pep_len=None,
    ):
        condition_memory = self.condition_encoder(
            src,
            src_key_padding_mask=src_mask,
            segment_ids=segment_ids,
            esm_hla_tokens=esm_hla_src,
            esm_hla_mask=esm_hla_mask,
            esm_pep_tokens=esm_pep_src,
            esm_pep_mask=esm_pep_mask,
            hla_lens=hla_len,
            pep_lens=pep_len,
        )
        condition = self._pool(condition_memory, src_mask)
        target_memory = self.target_encoder(
            tgt, src_key_padding_mask=tgt_pad_mask
        )
        target = self._pool(target_memory, tgt_pad_mask)

        posterior_input = torch.cat([condition, target], dim=1)
        mu_post = self.post_net_mu(posterior_input)
        logvar_post = self.post_net_logvar(posterior_input)
        mu_prior = self.prior_net_mu(condition)
        logvar_prior = self.prior_net_logvar(condition)

        latent = self.z_projector(
            self.reparameterize(mu_post, logvar_post)
        ).unsqueeze(1)
        decoder_memory = torch.cat([latent, condition_memory], dim=1)
        latent_mask = torch.zeros(
            (src.size(0), 1), dtype=torch.bool, device=src.device
        )
        memory_mask = torch.cat([latent_mask, src_mask], dim=1)
        logits = self.decoder(
            tgt,
            decoder_memory,
            tgt_mask=tgt_causal_mask,
            tgt_pad_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_mask,
        )
        return {
            "logits": logits,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
        }
