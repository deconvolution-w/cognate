from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .core.batching import build_condition_task_a, collate_items
from .core.utils import causal_mask
from .hla import load_hla_lookup, resolve_hla_sequence
from .model import ModelBundle
from .schemas import first_existing_column, read_csv

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def valid_cdr3(cdr3: str, min_len: int = 10, max_len: int = 20) -> bool:
    text = str(cdr3).strip().upper()
    return (
        min_len <= len(text) <= max_len
        and set(text) <= CANONICAL_AA
        and text.startswith("C")
        and text.endswith(("F", "W"))
    )


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _condition_batch_many(
    bundle: ModelBundle, hla_sequences: list[str], peptides: list[str]
) -> dict:
    if len(hla_sequences) != len(peptides):
        raise ValueError("hla_sequences and peptides must have the same length")
    esm_tokenizer = bundle.get_esm_tokenizer()
    items = []
    for hla_sequence, peptide in zip(hla_sequences, peptides):
        src, segments, hla_len, peptide_len = build_condition_task_a(
            bundle.tokenizer, hla_sequence, peptide
        )
        esm_hla = esm_peptide = None
        if esm_tokenizer is not None:
            esm_hla = esm_tokenizer(hla_sequence, return_tensors="pt")[
                "input_ids"
            ].squeeze(0)
            esm_peptide = esm_tokenizer(peptide, return_tensors="pt")[
                "input_ids"
            ].squeeze(0)
        items.append(
            {
                "src": src,
                "seg": segments,
                "tgt": torch.tensor([2, 3], dtype=torch.long),
                "hla_len": hla_len,
                "pep_len": peptide_len,
                "esm_hla": esm_hla,
                "esm_pep": esm_peptide,
            }
        )
    return collate_items(items, bundle.device, esm_tokenizer)


@torch.no_grad()
def _encode_condition(bundle: ModelBundle, hla_sequence: str, peptide: str):
    return _encode_condition_many(bundle, [hla_sequence], [peptide])


@torch.no_grad()
def _encode_condition_many(
    bundle: ModelBundle, hla_sequences: list[str], peptides: list[str]
):
    batch = _condition_batch_many(bundle, hla_sequences, peptides)
    model = bundle.model
    memory = model.condition_encoder(
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
    pooled = model._pool(memory, batch["src_mask"])
    return (
        memory,
        batch["src_mask"],
        model.prior_net_mu(pooled),
        model.prior_net_logvar(pooled),
    )


def _apply_repetition_penalty(logits, generated, penalty: float):
    if penalty == 1.0:
        return logits
    seen = torch.zeros_like(logits, dtype=torch.bool)
    seen.scatter_(1, generated, True)
    seen[:, 2] = False
    seen[:, 3] = False
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)


def _top_p_filter(logits, top_p: float):
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[:, 0] = False
    remove = remove.scatter(1, sorted_indices, remove)
    return logits.masked_fill(remove, float("-inf"))


@torch.no_grad()
def _beam_decode(
    bundle: ModelBundle,
    latent,
    memory,
    memory_mask,
    beam_width: int,
    num_sequences: int,
    max_len: int,
):
    if num_sequences > beam_width:
        raise ValueError("beam_width must be at least num_return_sequences")

    model = bundle.model
    batch_size = latent.size(0)
    latent_memory = model.z_projector(latent).unsqueeze(1)
    latent_memory = latent_memory.repeat_interleave(beam_width, dim=0)
    memory = memory.repeat_interleave(beam_width, dim=0)
    memory_mask = memory_mask.repeat_interleave(beam_width, dim=0)
    decoder_memory = torch.cat([latent_memory, memory], dim=1)
    decoder_mask = torch.cat(
        [
            torch.zeros(
                (decoder_memory.size(0), 1),
                dtype=torch.bool,
                device=decoder_memory.device,
            ),
            memory_mask,
        ],
        dim=1,
    )

    scores = torch.zeros(batch_size * beam_width, device=decoder_memory.device)
    for row in range(batch_size):
        scores[row * beam_width + 1 : (row + 1) * beam_width] = -1e9
    generated = torch.full(
        (batch_size * beam_width, 1),
        2,
        dtype=torch.long,
        device=decoder_memory.device,
    )
    finished = torch.zeros(
        batch_size * beam_width, dtype=torch.bool, device=decoder_memory.device
    )

    for _ in range(max_len):
        logits = model.decoder(
            generated,
            decoder_memory,
            tgt_mask=causal_mask(generated.size(1), generated.device),
            memory_key_padding_mask=decoder_mask,
        )[:, -1]
        vocab_size = logits.size(-1)
        candidates = (scores.unsqueeze(1) + F.log_softmax(logits, dim=-1)).view(
            batch_size, -1
        )
        next_scores, indices = candidates.topk(beam_width, dim=1)
        parent = indices.div(vocab_size, rounding_mode="floor")
        token = indices % vocab_size
        batch_offset = (
            torch.arange(batch_size, device=generated.device).unsqueeze(1) * beam_width
        )
        parent = (batch_offset + parent).reshape(-1)
        generated = torch.cat(
            [generated[parent], token.reshape(-1, 1)], dim=1
        )
        finished = finished[parent] | token.reshape(-1).eq(3)
        scores = next_scores.reshape(-1)
        if finished.all():
            break

    sequences = generated.view(batch_size, beam_width, -1)
    top_beams = scores.view(batch_size, beam_width).topk(num_sequences, dim=1).indices
    gather = top_beams.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
    return torch.gather(sequences, 1, gather).reshape(
        batch_size * num_sequences, -1
    )


@torch.no_grad()
def _nucleus_decode(
    bundle: ModelBundle,
    latent,
    memory,
    memory_mask,
    max_len: int,
    temperature: float = 1.0,
    top_p: float = 0.85,
    repetition_penalty: float = 1.2,
):
    model = bundle.model
    latent_memory = model.z_projector(latent).unsqueeze(1)
    decoder_memory = torch.cat([latent_memory, memory], dim=1)
    decoder_mask = torch.cat(
        [
            torch.zeros(
                (decoder_memory.size(0), 1),
                dtype=torch.bool,
                device=decoder_memory.device,
            ),
            memory_mask,
        ],
        dim=1,
    )

    generated = torch.full(
        (decoder_memory.size(0), 1), 2, dtype=torch.long, device=decoder_memory.device
    )
    finished = torch.zeros(
        decoder_memory.size(0),
        dtype=torch.bool,
        device=decoder_memory.device,
    )
    for _ in range(max_len):
        logits = model.decoder(
            generated,
            decoder_memory,
            tgt_mask=causal_mask(generated.size(1), generated.device),
            memory_key_padding_mask=decoder_mask,
        )[:, -1]
        logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
        logits = _top_p_filter(logits / temperature, top_p)
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        next_token[finished] = 3
        generated = torch.cat([generated, next_token], dim=1)
        finished |= next_token.squeeze(1).eq(3)
        if finished.all():
            break
    return generated


@torch.no_grad()
def _decode_fixed_latent(
    bundle: ModelBundle,
    latent,
    memory,
    memory_mask,
    num_sequences: int,
    max_len: int,
    temperature: float = 1.0,
    top_p: float = 0.85,
    repetition_penalty: float = 1.2,
):
    return _nucleus_decode(
        bundle,
        latent.repeat_interleave(num_sequences, dim=0),
        memory.repeat_interleave(num_sequences, dim=0),
        memory_mask.repeat_interleave(num_sequences, dim=0),
        max_len=max_len,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


@torch.no_grad()
def _decode_latent(
    bundle: ModelBundle,
    mean,
    log_variance,
    memory,
    memory_mask,
    num_sequences: int,
    max_len: int,
    temperature: float = 1.0,
    top_p: float = 0.85,
    repetition_penalty: float = 1.2,
):
    # Independent latent draws preserve nucleus-sampling diversity.
    mean = mean.repeat_interleave(num_sequences, dim=0)
    log_variance = log_variance.repeat_interleave(num_sequences, dim=0)
    latent = bundle.model.reparameterize(mean, log_variance)
    return _nucleus_decode(
        bundle,
        latent,
        memory.repeat_interleave(num_sequences, dim=0),
        memory_mask.repeat_interleave(num_sequences, dim=0),
        max_len=max_len,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


@torch.no_grad()
def generate_cdr3_for_condition(
    bundle: ModelBundle,
    hla_sequence: str,
    peptide: str,
    num_return_sequences: int = 1,
    strategy: str = "beam",
    beam_width: int = 5,
    max_len: int = 25,
    temperature: float = 1.0,
    top_p: float = 0.85,
    repetition_penalty: float = 1.2,
    seed: int | None = 3407,
    cdr3_min_len: int = 10,
    cdr3_max_len: int = 20,
    keep_invalid: bool = False,
) -> list[str]:
    strategy = strategy.strip().lower().replace("_sampling", "")
    if strategy == "beam_search":
        strategy = "beam"
    if strategy not in {"beam", "nucleus"}:
        raise ValueError("strategy must be 'beam' or 'nucleus'")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be positive")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Require temperature > 0 and 0 < top_p <= 1")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    _set_seed(seed)
    memory, mask, mean, log_variance = _encode_condition(
        bundle, hla_sequence, peptide
    )
    if strategy == "beam":
        decoded = _beam_decode(
            bundle,
            mean,
            memory,
            mask,
            beam_width,
            num_return_sequences,
            max_len,
        )
    else:
        decoded = _decode_latent(
            bundle,
            mean,
            log_variance,
            memory,
            mask,
            num_return_sequences,
            max_len,
            temperature,
            top_p,
            repetition_penalty,
        )
    sequences = [
        bundle.tokenizer.decode(row, skip_special_tokens=True).strip().upper()
        for row in decoded
    ]
    if not keep_invalid:
        sequences = [
            sequence
            for sequence in sequences
            if valid_cdr3(sequence, cdr3_min_len, cdr3_max_len)
        ]
    return list(dict.fromkeys(sequences))


@torch.no_grad()
def generate_cdr3_for_conditions(
    bundle: ModelBundle,
    hla_sequences: list[str],
    peptides: list[str],
    *,
    num_return_sequences: int = 1,
    strategy: str = "beam",
    beam_width: int = 5,
    max_len: int = 25,
    temperature: float = 1.0,
    top_p: float = 0.85,
    repetition_penalty: float = 1.2,
    seed: int | None = 3407,
    batch_size: int = 32,
    keep_invalid: bool = False,
) -> list[list[str]]:
    if len(hla_sequences) != len(peptides):
        raise ValueError("hla_sequences and peptides must have the same length")
    strategy = strategy.strip().lower()
    if strategy not in {"beam", "nucleus"}:
        raise ValueError("strategy must be 'beam' or 'nucleus'")
    _set_seed(seed)
    outputs: list[list[str]] = []
    for start in range(0, len(peptides), batch_size):
        stop = min(start + batch_size, len(peptides))
        memory, mask, mean, log_variance = _encode_condition_many(
            bundle, hla_sequences[start:stop], peptides[start:stop]
        )
        if strategy == "beam":
            decoded = _beam_decode(
                bundle,
                mean,
                memory,
                mask,
                beam_width,
                num_return_sequences,
                max_len,
            )
        else:
            decoded = _decode_latent(
                bundle,
                mean,
                log_variance,
                memory,
                mask,
                num_return_sequences,
                max_len,
                temperature,
                top_p,
                repetition_penalty,
            )
        decoded = decoded.view(stop - start, num_return_sequences, -1)
        for condition_rows in decoded:
            sequences = [
                bundle.tokenizer.decode(row, skip_special_tokens=True).strip().upper()
                for row in condition_rows
            ]
            if not keep_invalid:
                sequences = [sequence for sequence in sequences if valid_cdr3(sequence)]
            outputs.append(sequences)
    return outputs


def generate_dataframe(
    bundle: ModelBundle,
    table: pd.DataFrame,
    hla_lookup_path: str | Path | None = None,
    seed: int | None = 3407,
    **generation_options,
) -> pd.DataFrame:
    peptide_column = first_existing_column(table, "peptide")
    sample_column = first_existing_column(table, "sample_id", required=False)
    hla_lookup = load_hla_lookup(hla_lookup_path)
    rows = []
    for offset, (index, row) in enumerate(table.iterrows()):
        peptide = str(row[peptide_column]).strip().upper()
        hla_sequence = resolve_hla_sequence(row, hla_lookup)
        sample_id = str(row[sample_column]).strip() if sample_column else str(index)
        row_seed = None if seed is None else seed + offset
        strategy = str(generation_options.get("strategy", "beam")).strip().lower()
        sequences = generate_cdr3_for_condition(
            bundle,
            hla_sequence,
            peptide,
            seed=row_seed,
            **generation_options,
        )
        for rank, cdr3 in enumerate(sequences, start=1):
            output_row = {
                "sample_id": sample_id,
                "peptide": peptide,
                "hla_sequence": hla_sequence,
                "generation_strategy": strategy,
                "generation_seed": row_seed if row_seed is not None else "random",
                "num_return_sequences": generation_options.get(
                    "num_return_sequences", 1
                ),
                "max_len": generation_options.get("max_len", 25),
                "cdr3_min_len": generation_options.get("cdr3_min_len", 10),
                "cdr3_max_len": generation_options.get("cdr3_max_len", 20),
                "keep_invalid": generation_options.get("keep_invalid", False),
                "rank": rank,
                "cdr3": cdr3,
            }
            if strategy == "beam":
                output_row["beam_width"] = generation_options.get("beam_width", 5)
            else:
                output_row.update(
                    {
                        "temperature": generation_options.get("temperature", 1.0),
                        "top_p": generation_options.get("top_p", 0.85),
                        "repetition_penalty": generation_options.get(
                            "repetition_penalty", 1.2
                        ),
                    }
                )
            rows.append(output_row)
    strategy = str(generation_options.get("strategy", "beam")).strip().lower()
    strategy_columns = (
        ["beam_width"]
        if strategy == "beam"
        else ["temperature", "top_p", "repetition_penalty"]
    )
    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "peptide",
            "hla_sequence",
            "generation_strategy",
            "generation_seed",
            "num_return_sequences",
            "max_len",
            *strategy_columns,
            "cdr3_min_len",
            "cdr3_max_len",
            "keep_invalid",
            "rank",
            "cdr3",
        ],
    )


def generate_csv(
    bundle: ModelBundle,
    input_csv: str | Path,
    output_csv: str | Path,
    max_rows: int | None = None,
    **options,
) -> pd.DataFrame:
    result = generate_dataframe(
        bundle, read_csv(input_csv, max_rows=max_rows), **options
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
