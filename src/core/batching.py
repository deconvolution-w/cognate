from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import (
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    TAG_CDR3,
    TAG_EPI,
    TAG_HLA,
    UNK_IDX,
    AminoAcidTokenizer,
)


def _encode(tokenizer: AminoAcidTokenizer, sequence: str) -> list[int]:
    return tokenizer.encode(sequence, add_special_tokens=False)


def build_condition_task_a(
    tokenizer: AminoAcidTokenizer, hla: str, peptide: str
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    hla_ids = _encode(tokenizer, hla)
    peptide_ids = _encode(tokenizer, peptide)
    values = [
        tokenizer.token2id[TAG_HLA],
        *hla_ids,
        tokenizer.token2id[TAG_EPI],
        *peptide_ids,
    ]
    segments = [2, *([0] * len(hla_ids)), 2, *([1] * len(peptide_ids))]
    return (
        torch.tensor(values, dtype=torch.long),
        torch.tensor(segments, dtype=torch.long),
        len(hla_ids),
        len(peptide_ids),
    )


def build_condition_task_b(
    tokenizer: AminoAcidTokenizer, hla: str, cdr3: str
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    hla_ids = _encode(tokenizer, hla)
    cdr3_ids = _encode(tokenizer, cdr3)
    values = [
        tokenizer.token2id[TAG_HLA],
        *hla_ids,
        tokenizer.token2id[TAG_CDR3],
        *cdr3_ids,
    ]
    segments = [2, *([0] * len(hla_ids)), 2, *([1] * len(cdr3_ids))]
    return (
        torch.tensor(values, dtype=torch.long),
        torch.tensor(segments, dtype=torch.long),
        len(hla_ids),
        len(cdr3_ids),
    )


def build_target(tokenizer: AminoAcidTokenizer, sequence: str) -> torch.Tensor:
    return torch.tensor(
        [SOS_IDX, *_encode(tokenizer, sequence), EOS_IDX], dtype=torch.long
    )


def collate_items(
    items: list[dict[str, Any]],
    device: torch.device | str,
    protein_tokenizer=None,
) -> dict[str, Any]:
    src = pad_sequence([item["src"] for item in items], True, padding_value=PAD_IDX)
    seg = pad_sequence([item["seg"] for item in items], True, padding_value=2)
    tgt = pad_sequence([item["tgt"] for item in items], True, padding_value=PAD_IDX)
    batch: dict[str, Any] = {
        "src": src.to(device),
        "tgt": tgt.to(device),
        "src_mask": (src == PAD_IDX).to(device),
        "tgt_pad_mask": (tgt == PAD_IDX).to(device),
        "segment_ids": seg.to(device),
        "hla_len": torch.tensor(
            [item["hla_len"] for item in items], dtype=torch.long, device=device
        ),
        "pep_len": torch.tensor(
            [item["pep_len"] for item in items], dtype=torch.long, device=device
        ),
    }
    if protein_tokenizer is not None and items[0].get("esm_hla") is not None:
        pad_id = protein_tokenizer.pad_token_id
        esm_hla = pad_sequence(
            [item["esm_hla"] for item in items], True, padding_value=pad_id
        )
        esm_pep = pad_sequence(
            [item["esm_pep"] for item in items], True, padding_value=pad_id
        )
        batch.update(
            {
                "esm_hla_src": esm_hla.to(device),
                "esm_pep_src": esm_pep.to(device),
                "esm_hla_mask": (esm_hla == pad_id).to(device),
                "esm_pep_mask": (esm_pep == pad_id).to(device),
            }
        )
    return batch


class ReciprocalCollator:
    def __init__(
        self,
        tokenizer: AminoAcidTokenizer,
        protein_tokenizer=None,
        training: bool = False,
        mask_probability: float = 0.0,
        hla_include_probability: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.training = training
        self.mask_probability = float(mask_probability)
        self.hla_include_probability = float(hla_include_probability)

    def _noisy(self, values: Iterable[int]) -> list[int]:
        if not self.training or self.mask_probability <= 0:
            return list(values)
        return [
            UNK_IDX if torch.rand(()) < self.mask_probability else int(value)
            for value in values
        ]

    def _include_hla(self) -> bool:
        if not self.training:
            return True
        return bool(torch.rand(()) < self.hla_include_probability)

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
        task_a: list[dict[str, Any]] = []
        task_b: list[dict[str, Any]] = []
        for row in rows:
            cdr3 = row["cdr3"].strip().upper()
            peptide = row["peptide"].strip().upper()
            full_hla = row["hla"].strip().upper()
            hla = full_hla if self._include_hla() else ""
            hla_ids = self._noisy(_encode(self.tokenizer, hla))
            peptide_ids = self._noisy(_encode(self.tokenizer, peptide))
            cdr3_ids = self._noisy(_encode(self.tokenizer, cdr3))

            def make_item(context_ids, context_tag, target, context_text):
                values = [
                    self.tokenizer.token2id[TAG_HLA],
                    *hla_ids,
                    context_tag,
                    *context_ids,
                ]
                segments = [
                    2,
                    *([0] * len(hla_ids)),
                    2,
                    *([1] * len(context_ids)),
                ]
                esm_hla = esm_context = None
                if self.protein_tokenizer is not None:
                    esm_hla = self.protein_tokenizer(hla, return_tensors="pt")[
                        "input_ids"
                    ].squeeze(0)
                    esm_context = self.protein_tokenizer(
                        context_text, return_tensors="pt"
                    )["input_ids"].squeeze(0)
                return {
                    "src": torch.tensor(values, dtype=torch.long),
                    "seg": torch.tensor(segments, dtype=torch.long),
                    "tgt": build_target(self.tokenizer, target),
                    "hla_len": len(hla_ids),
                    "pep_len": len(context_ids),
                    "esm_hla": esm_hla,
                    "esm_pep": esm_context,
                }

            task_a.append(
                make_item(
                    peptide_ids, self.tokenizer.token2id[TAG_EPI], cdr3, peptide
                )
            )
            task_b.append(
                make_item(
                    cdr3_ids, self.tokenizer.token2id[TAG_CDR3], peptide, cdr3
                )
            )
        return {
            "task_a": collate_items(task_a, "cpu", self.protein_tokenizer),
            "task_b": collate_items(task_b, "cpu", self.protein_tokenizer),
        }
