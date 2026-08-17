from __future__ import annotations

from pathlib import Path
from typing import Any

from .embeddings import embed_input
from .hla import load_hla_lookup, resolve_hla_sequence
from .inference import generate_cdr3_for_condition
from .model import ModelBundle, load_cognate
from .scoring import score_triplet
from .targeting import generate_multitarget as generate_multitarget_candidates
from .targeting import rank_target_pairs as rank_target_pair_table


class Cognate:
    def __init__(self, bundle: ModelBundle, hla_lookup: str | Path | None = None) -> None:
        self.bundle = bundle
        self.hla_lookup = load_hla_lookup(hla_lookup)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | None = None,
        config: str | Path | None = None,
        device: str | None = None,
        offline: bool = True,
        hla_lookup: str | Path | None = None,
    ) -> "Cognate":
        return cls(
            load_cognate(
                config_path=config,
                checkpoint_path=checkpoint,
                device=device,
                offline=offline,
            ),
            hla_lookup=hla_lookup,
        )

    def _hla(self, hla: str | None, hla_sequence: str | None) -> str:
        row = {"hla_sequence": hla_sequence or "", "hla": hla or ""}
        return resolve_hla_sequence(row, self.hla_lookup)

    def generate(
        self,
        peptide: str,
        hla: str | None = None,
        hla_sequence: str | None = None,
        n: int = 1,
        strategy: str = "beam",
        seed: int | None = 3407,
        **decoding: Any,
    ) -> list[str]:
        return generate_cdr3_for_condition(
            self.bundle,
            self._hla(hla, hla_sequence),
            peptide.strip().upper(),
            num_return_sequences=n,
            strategy=strategy,
            seed=seed,
            **decoding,
        )

    def score(
        self,
        peptide: str,
        cdr3: str,
        hla: str | None = None,
        hla_sequence: str | None = None,
        task_a_weight: float = 0.5,
        task_b_weight: float = 0.5,
    ) -> dict[str, float]:
        return score_triplet(
            self.bundle,
            self._hla(hla, hla_sequence),
            peptide.strip().upper(),
            cdr3.strip().upper(),
            task_a_weight=task_a_weight,
            task_b_weight=task_b_weight,
        )

    def embed(
        self,
        peptide: str | None = None,
        cdr3: str | None = None,
        hla: str | None = None,
        hla_sequence: str | None = None,
    ) -> dict[str, Any]:
        resolved_hla = (
            self._hla(hla, hla_sequence) if hla or hla_sequence else None
        )
        return embed_input(
            self.bundle,
            peptide=peptide,
            cdr3=cdr3,
            hla_sequence=resolved_hla,
        )

    def rank_target_pairs(
        self,
        table,
        calibration_column: str = "allowed_for_calibration",
    ):
        return rank_target_pair_table(
            table,
            calibration_column=calibration_column,
        )

    def generate_multitarget(
        self,
        targets: list[str],
        hla: str | None = None,
        hla_sequence: str | None = None,
        off_targets: list[str] | None = None,
        method: str = "overlap",
        n: int = 10,
        pool_size: int = 100,
        seed: int = 3407,
        **options: Any,
    ):
        return generate_multitarget_candidates(
            self.bundle,
            self._hla(hla, hla_sequence),
            targets,
            off_targets,
            method=method,
            n=n,
            pool_size=pool_size,
            seed=seed,
            **options,
        )
