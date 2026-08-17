from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .core.batching import (
    build_condition_task_a,
    build_condition_task_b,
    build_target,
    collate_items,
)
from .hla import load_hla_lookup, resolve_hla_sequence
from .model import ModelBundle
from .schemas import first_existing_column, read_csv


def _condition_item(
    bundle: ModelBundle,
    direction: str,
    hla_sequence: str,
    payload: str,
    use_esm: bool,
) -> dict:
    if direction == "task_a":
        src, seg, hla_len, payload_len = build_condition_task_a(
            bundle.tokenizer, hla_sequence, payload
        )
    elif direction == "task_b":
        src, seg, hla_len, payload_len = build_condition_task_b(
            bundle.tokenizer, hla_sequence, payload
        )
    else:
        raise ValueError("direction must be 'task_a' or 'task_b'")

    esm_hla = esm_payload = None
    esm_tokenizer = bundle.get_esm_tokenizer() if use_esm else None
    if esm_tokenizer is not None:
        esm_hla = esm_tokenizer(hla_sequence, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)
        esm_payload = esm_tokenizer(payload, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)

    return {
        "src": src,
        "seg": seg,
        "tgt": build_target(bundle.tokenizer, ""),
        "hla_len": hla_len,
        "pep_len": payload_len,
        "esm_hla": esm_hla,
        "esm_pep": esm_payload,
    }


@torch.no_grad()
def encode_conditions(
    bundle: ModelBundle,
    hla_sequences: Sequence[str],
    payloads: Sequence[str],
    direction: str,
    *,
    use_esm: bool,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if len(hla_sequences) != len(payloads):
        raise ValueError("hla_sequences and payloads must have the same length")
    if not payloads:
        return np.empty((0, 640), np.float32), np.empty((0, 96), np.float32)

    pooled_parts: list[np.ndarray] = []
    prior_parts: list[np.ndarray] = []
    esm_tokenizer = bundle.get_esm_tokenizer() if use_esm else None

    for start in range(0, len(payloads), batch_size):
        stop = min(start + batch_size, len(payloads))
        items = [
            _condition_item(
                bundle,
                direction,
                str(hla_sequences[index]).strip().upper(),
                str(payloads[index]).strip().upper(),
                use_esm,
            )
            for index in range(start, stop)
        ]
        batch = collate_items(items, bundle.device, esm_tokenizer)
        memory = bundle.model.condition_encoder(
            batch["src"],
            src_key_padding_mask=batch["src_mask"],
            segment_ids=batch["segment_ids"],
            esm_hla_tokens=batch.get("esm_hla_src") if use_esm else None,
            esm_hla_mask=batch.get("esm_hla_mask") if use_esm else None,
            esm_pep_tokens=batch.get("esm_pep_src") if use_esm else None,
            esm_pep_mask=batch.get("esm_pep_mask") if use_esm else None,
            hla_lens=batch.get("hla_len"),
            pep_lens=batch.get("pep_len"),
        )
        pooled = bundle.model._pool(memory, batch["src_mask"])
        prior = bundle.model.prior_net_mu(pooled)
        pooled_parts.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        prior_parts.append(prior.detach().cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(pooled_parts), np.concatenate(prior_parts)


@torch.no_grad()
def embed_pmhc(
    bundle: ModelBundle, hla_sequence: str, peptide: str
) -> dict[str, np.ndarray]:
    h_pmhc, mu_pmhc = encode_conditions(
        bundle, [hla_sequence], [peptide], "task_a", use_esm=True, batch_size=1
    )
    pmhc_block = np.concatenate([h_pmhc, mu_pmhc], axis=1)
    return {
        "pmhc_condition_640": h_pmhc[0],
        "pmhc_prior_96": mu_pmhc[0],
        "pmhc_block_736": pmhc_block[0],
    }


@torch.no_grad()
def embed_peptide(bundle: ModelBundle, peptide: str) -> dict[str, np.ndarray]:
    h_peptide, mu_peptide = encode_conditions(
        bundle, [""], [peptide], "task_a", use_esm=True, batch_size=1
    )
    peptide_block = np.concatenate([h_peptide, mu_peptide], axis=1)
    return {
        "peptide_condition_640": h_peptide[0],
        "peptide_prior_96": mu_peptide[0],
        "peptide_block_736": peptide_block[0],
    }


@torch.no_grad()
def embed_cdr3(bundle: ModelBundle, cdr3: str) -> dict[str, np.ndarray]:
    h_tcr, mu_tcr = encode_conditions(
        bundle, [""], [cdr3], "task_b", use_esm=True, batch_size=1
    )
    tcr_block = np.concatenate([h_tcr, mu_tcr], axis=1)
    return {
        "tcr_condition_640": h_tcr[0],
        "tcr_prior_96": mu_tcr[0],
        "tcr_block_736": tcr_block[0],
    }


@torch.no_grad()
def embed_triplet(
    bundle: ModelBundle, hla_sequence: str, peptide: str, cdr3: str
) -> dict[str, np.ndarray]:
    pmhc = embed_pmhc(bundle, hla_sequence, peptide)
    tcr = embed_cdr3(bundle, cdr3)
    return {
        **pmhc,
        **tcr,
        "full_1472": np.concatenate(
            [pmhc["pmhc_block_736"], tcr["tcr_block_736"]]
        ),
    }


@torch.no_grad()
def embed_empty_hla_pair(
    bundle: ModelBundle, peptide: str, cdr3: str
) -> dict[str, np.ndarray]:
    peptide_features = embed_peptide(bundle, peptide)
    tcr_features = embed_cdr3(bundle, cdr3)
    return {
        **peptide_features,
        **tcr_features,
        "empty_hla_full_1472": np.concatenate(
            [
                peptide_features["peptide_block_736"],
                tcr_features["tcr_block_736"],
            ]
        ),
    }


def embed_input(
    bundle: ModelBundle,
    *,
    peptide: str | None = None,
    cdr3: str | None = None,
    hla_sequence: str | None = None,
) -> dict[str, np.ndarray]:
    peptide = str(peptide).strip().upper() if peptide is not None else ""
    cdr3 = str(cdr3).strip().upper() if cdr3 is not None else ""
    hla_sequence = (
        str(hla_sequence).strip().upper() if hla_sequence is not None else ""
    )
    if not peptide and not cdr3:
        raise ValueError("Provide peptide, cdr3, or both.")
    if hla_sequence and not peptide:
        raise ValueError("HLA is only embedded together with a peptide.")
    if peptide and cdr3:
        if hla_sequence:
            return embed_triplet(bundle, hla_sequence, peptide, cdr3)
        return embed_empty_hla_pair(bundle, peptide, cdr3)
    if peptide:
        if hla_sequence:
            return embed_pmhc(bundle, hla_sequence, peptide)
        return embed_peptide(bundle, peptide)
    return embed_cdr3(bundle, cdr3)


def _expand_condition_rows(
    bundle: ModelBundle,
    hla_sequences: list[str],
    payloads: list[str],
    direction: str,
    *,
    use_esm: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = list(zip(hla_sequences, payloads))
    unique = list(dict.fromkeys(keys))
    index = {key: row for row, key in enumerate(unique)}
    pooled, prior = encode_conditions(
        bundle,
        [key[0] for key in unique],
        [key[1] for key in unique],
        direction,
        use_esm=use_esm,
        batch_size=batch_size,
    )
    rows = np.fromiter((index[key] for key in keys), dtype=np.int64)
    return pooled[rows], prior[rows]


def embed_dataframe(
    bundle: ModelBundle,
    table: pd.DataFrame,
    npz_path: str | Path,
    metadata_path: str | Path,
    hla_lookup_path: str | Path | None = None,
    batch_size: int = 128,
) -> tuple[Path, Path]:
    peptide_col = first_existing_column(table, "peptide", required=False)
    cdr3_col = first_existing_column(table, "cdr3", required=False)
    sample_col = first_existing_column(table, "sample_id", required=False)
    if peptide_col is None and cdr3_col is None:
        raise ValueError("Input requires a peptide or cdr3 column.")

    hla_col = first_existing_column(table, "hla", required=False)
    hla_sequence_col = first_existing_column(table, "hla_sequence", required=False)
    has_hla = hla_col is not None or hla_sequence_col is not None
    if has_hla and peptide_col is None:
        raise ValueError("HLA is only embedded together with a peptide column.")

    arrays: dict[str, np.ndarray] = {}
    metadata_values: dict[str, object] = {
        "array_index": np.arange(len(table)),
        "sample_id": (
            table[sample_col].astype(str).tolist()
            if sample_col
            else table.index.astype(str).tolist()
        ),
    }
    h_antigen = mu_antigen = h_tcr = mu_tcr = None
    if peptide_col is not None:
        peptides = table[peptide_col].astype(str).str.strip().str.upper().tolist()
        if has_hla:
            hla_lookup = load_hla_lookup(hla_lookup_path)
            hla_sequences = [
                resolve_hla_sequence(row, hla_lookup)
                for _, row in table.iterrows()
            ]
            prefix = "pmhc"
        else:
            hla_sequences = [""] * len(peptides)
            prefix = "peptide"
        h_antigen, mu_antigen = _expand_condition_rows(
            bundle,
            hla_sequences,
            peptides,
            "task_a",
            use_esm=True,
            batch_size=batch_size,
        )
        arrays.update(
            {
                f"{prefix}_condition_640": h_antigen,
                f"{prefix}_prior_96": mu_antigen,
                f"{prefix}_block_736": np.concatenate(
                    [h_antigen, mu_antigen], axis=1
                ),
            }
        )
        if has_hla:
            metadata_values["hla_sequence"] = hla_sequences
        metadata_values["peptide"] = peptides

    if cdr3_col is not None:
        cdr3s = table[cdr3_col].astype(str).str.strip().str.upper().tolist()
        h_tcr, mu_tcr = _expand_condition_rows(
            bundle,
            [""] * len(cdr3s),
            cdr3s,
            "task_b",
            use_esm=True,
            batch_size=batch_size,
        )
        arrays.update(
            {
                "tcr_condition_640": h_tcr,
                "tcr_prior_96": mu_tcr,
                "tcr_block_736": np.concatenate([h_tcr, mu_tcr], axis=1),
            }
        )
        metadata_values["cdr3"] = cdr3s

    if peptide_col is not None and cdr3_col is not None:
        representation = "full_1472" if has_hla else "empty_hla_full_1472"
        arrays[representation] = np.concatenate(
            [h_antigen, mu_antigen, h_tcr, mu_tcr], axis=1
        )
        dimension = 1472
    elif peptide_col is not None:
        representation = "pmhc_block_736" if has_hla else "peptide_block_736"
        dimension = 736
    else:
        representation, dimension = "tcr_block_736", 736
    metadata_values.update(
        {
            "representation": [representation] * len(table),
            "embedding_dimension": [dimension] * len(table),
            "batch_size": [batch_size] * len(table),
        }
    )
    metadata = pd.DataFrame(metadata_values)
    npz_output = Path(npz_path)
    metadata_output = Path(metadata_path)
    npz_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_output, **arrays)
    metadata.to_csv(metadata_output, index=False)
    return npz_output, metadata_output


def embed_csv(
    bundle: ModelBundle,
    input_csv: str | Path,
    npz_path: str | Path,
    metadata_path: str | Path,
    max_rows: int | None = None,
    **kwargs,
) -> tuple[Path, Path]:
    table = read_csv(input_csv, max_rows=max_rows)
    return embed_dataframe(bundle, table, npz_path, metadata_path, **kwargs)
