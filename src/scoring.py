from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .hla import load_hla_lookup, resolve_hla_sequence
from .model import ModelBundle
from .core.batching import (
    build_condition_task_a,
    build_condition_task_b,
    build_target,
    collate_items,
)
from .schemas import first_existing_column, read_csv


def _make_direction_batch(
    bundle: ModelBundle,
    hla_sequence: str,
    context_sequence: str,
    target_sequence: str,
    direction: str,
) -> dict:
    if direction == "task_a":
        src, seg, hla_len, pep_len = build_condition_task_a(
            bundle.tokenizer, hla_sequence, context_sequence
        )
    elif direction == "task_b":
        src, seg, hla_len, pep_len = build_condition_task_b(
            bundle.tokenizer, hla_sequence, context_sequence
        )
    else:
        raise ValueError("direction must be task_a or task_b")

    tgt = build_target(bundle.tokenizer, target_sequence)
    esm_tokenizer = bundle.get_esm_tokenizer()
    esm_hla = esm_context = None
    if esm_tokenizer is not None:
        esm_hla = esm_tokenizer(hla_sequence, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)
        esm_context = esm_tokenizer(context_sequence, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)
    item = {
        "src": src,
        "seg": seg,
        "tgt": tgt,
        "hla_len": hla_len,
        "pep_len": pep_len,
        "esm_hla": esm_hla,
        "esm_pep": esm_context,
    }
    return collate_items([item], bundle.device, esm_tokenizer)


@torch.no_grad()
def _deterministic_direction_nll(
    bundle: ModelBundle,
    hla_sequence: str,
    context_sequence: str,
    target_sequence: str,
    direction: str,
) -> float:
    """Teacher-forced NLL using posterior mean instead of posterior sampling."""
    batch = _make_direction_batch(
        bundle, hla_sequence, context_sequence, target_sequence, direction
    )
    model = bundle.model
    condition_memory = model.condition_encoder(
        batch["src"],
        src_key_padding_mask=batch["src_mask"],
        segment_ids=batch["segment_ids"],
        esm_hla_tokens=batch.get("esm_hla_src"),
        esm_hla_mask=batch.get("esm_hla_mask"),
        esm_pep_tokens=batch.get("esm_pep_src"),
        esm_pep_mask=batch.get("esm_pep_mask"),
        hla_lens=batch.get("hla_len"),
        pep_lens=batch.get("pep_len"),
    )
    condition_summary = model._pool(condition_memory, batch["src_mask"])
    target_memory = model.target_encoder(
        batch["tgt"], src_key_padding_mask=batch["tgt_pad_mask"]
    )
    target_summary = model._pool(target_memory, batch["tgt_pad_mask"])
    posterior_mean = model.post_net_mu(
        torch.cat([condition_summary, target_summary], dim=1)
    )
    latent_embedding = model.z_projector(posterior_mean).unsqueeze(1)

    full_memory = torch.cat([latent_embedding, condition_memory], dim=1)
    latent_mask = torch.zeros(
        (batch["src"].size(0), 1),
        dtype=torch.bool,
        device=batch["src"].device,
    )
    full_memory_mask = torch.cat([latent_mask, batch["src_mask"]], dim=1)
    logits = model.decoder(
        batch["tgt"],
        full_memory,
        tgt_pad_mask=batch["tgt_pad_mask"],
        memory_key_padding_mask=full_memory_mask,
    )

    logits_shifted = logits[:, :-1, :]
    tgt_shifted = batch["tgt"][:, 1:]
    pad_shifted = batch["tgt_pad_mask"][:, 1:]
    ce = F.cross_entropy(
        logits_shifted.reshape(-1, logits_shifted.size(-1)),
        tgt_shifted.reshape(-1),
        reduction="none",
    ).reshape(tgt_shifted.shape)
    lengths = (~pad_shifted).float().sum(dim=1).clamp(min=1.0)
    nll = (ce * (~pad_shifted).float()).sum(dim=1) / lengths
    return float(nll.cpu().item())


@torch.no_grad()
def score_triplet(
    bundle: ModelBundle,
    hla_sequence: str,
    peptide: str,
    cdr3: str,
    task_a_weight: float = 0.5,
    task_b_weight: float = 0.5,
) -> dict:
    if task_a_weight < 0 or task_b_weight < 0:
        raise ValueError("task weights must be non-negative")
    if task_a_weight + task_b_weight == 0:
        raise ValueError("at least one task weight must be positive")
    nll_a = _deterministic_direction_nll(
        bundle,
        hla_sequence=hla_sequence,
        context_sequence=peptide,
        target_sequence=cdr3,
        direction="task_a",
    )
    nll_b = _deterministic_direction_nll(
        bundle,
        hla_sequence=hla_sequence,
        context_sequence=cdr3,
        target_sequence=peptide,
        direction="task_b",
    )
    score_a, score_b = -nll_a, -nll_b
    return {
        "score_task_a": score_a,
        "score_task_b": score_b,
        "score_pair_raw": task_a_weight * score_a + task_b_weight * score_b,
        "nll_task_a": nll_a,
        "nll_task_b": nll_b,
    }


def score_dataframe(
    bundle: ModelBundle,
    table: pd.DataFrame,
    hla_lookup_path: str | Path | None = None,
    task_a_weight: float = 0.5,
    task_b_weight: float = 0.5,
) -> pd.DataFrame:
    peptide_col = first_existing_column(table, "peptide")
    cdr3_col = first_existing_column(table, "cdr3")
    sample_col = first_existing_column(table, "sample_id", required=False)
    hla_lookup = load_hla_lookup(hla_lookup_path)
    rows = []
    for index, row in table.iterrows():
        hla_sequence = resolve_hla_sequence(row, hla_lookup)
        peptide = str(row[peptide_col]).strip().upper()
        cdr3 = str(row[cdr3_col]).strip().upper()
        scores = score_triplet(
            bundle,
            hla_sequence=hla_sequence,
            peptide=peptide,
            cdr3=cdr3,
            task_a_weight=task_a_weight,
            task_b_weight=task_b_weight,
        )
        rows.append(
            {
                "sample_id": str(row[sample_col]).strip() if sample_col else str(index),
                "hla_sequence": hla_sequence,
                "peptide": peptide,
                "cdr3": cdr3,
                "task_a_weight": task_a_weight,
                "task_b_weight": task_b_weight,
                **scores,
            }
        )
    return pd.DataFrame(rows)


def score_csv(
    bundle: ModelBundle,
    input_csv: str | Path,
    output_csv: str | Path,
    max_rows: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    table = read_csv(input_csv, max_rows=max_rows)
    result = score_dataframe(bundle, table, **kwargs)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
