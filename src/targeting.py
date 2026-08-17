from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .hla import load_hla_lookup, resolve_hla_sequence
from .inference import (
    _decode_fixed_latent,
    _encode_condition,
    _set_seed,
    generate_cdr3_for_condition,
    valid_cdr3,
)
from .model import ModelBundle
from .schemas import first_existing_column, read_csv
from .scoring import score_triplet


# Output name, coefficient, accepted z-score columns, accepted raw columns.
TARGET_PAIR_TERMS = (
    (
        "z_dual_target_margin",
        1.15,
        ("z_dual_target_margin",),
        ("dual_target_margin", "dual_target_score"),
    ),
    (
        "z_control_adjusted_margin",
        1.15,
        ("z_control_adjusted_margin",),
        ("control_adjusted_margin",),
    ),
    (
        "z_hla_robustness",
        0.05,
        ("z_hla_robustness", "z_HLA_robustness"),
        ("hla_robustness", "hla_robustness_score", "HLA_robustness_score"),
    ),
    (
        "z_peptide_tier_robustness",
        0.05,
        ("z_peptide_tier_robustness",),
        ("peptide_tier_robustness", "peptide_tier_robust_numeric"),
    ),
    (
        "z_benign_ligand_penalty",
        -0.85,
        ("z_benign_ligand_penalty",),
        ("benign_ligand_penalty", "normal_ligand_penalty"),
    ),
    (
        "z_hub_penalty",
        -0.75,
        ("z_hub_penalty",),
        ("hub_penalty",),
    ),
    (
        "z_overlap_dominance_penalty",
        -1.25,
        ("z_overlap_dominance_penalty",),
        ("overlap_dominance_penalty", "overlap_log"),
    ),
)


def _first_column(table: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in table.columns), None)


def _boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _pair_labels(table: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    pair_col = _first_column(table, ("target_pair", "pair_id", "pair"))
    target_a_col = _first_column(table, ("target_a", "target_A", "antigen_a"))
    target_b_col = _first_column(table, ("target_b", "target_B", "antigen_b"))

    target_a = (
        table[target_a_col].astype(str).str.strip()
        if target_a_col
        else pd.Series(["not_provided"] * len(table), index=table.index)
    )
    target_b = (
        table[target_b_col].astype(str).str.strip()
        if target_b_col
        else pd.Series(["not_provided"] * len(table), index=table.index)
    )
    if pair_col:
        pair = table[pair_col].astype(str).str.strip()
    elif target_a_col and target_b_col:
        pair = target_a + "/" + target_b
    else:
        pair = pd.Series([str(index) for index in table.index], index=table.index)
    return pair, target_a, target_b


def rank_target_pairs(
    table: pd.DataFrame,
    calibration_column: str = "allowed_for_calibration",
) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Target-pair input is empty.")

    result = pd.DataFrame(index=table.index)
    pair, target_a, target_b = _pair_labels(table)
    result["target_pair"] = pair
    if (target_a != "not_provided").any():
        result["target_a"] = target_a
    if (target_b != "not_provided").any():
        result["target_b"] = target_b

    z_columns = [_first_column(table, term[2]) for term in TARGET_PAIR_TERMS]
    raw_columns = [_first_column(table, term[3]) for term in TARGET_PAIR_TERMS]
    have_all_raw = all(column is not None for column in raw_columns)
    use_input_z = all(column is not None for column in z_columns) and not have_all_raw
    if not use_input_z and not have_all_raw:
        missing = [
            output_name
            for (output_name, _, _, _), z_col, raw_col in zip(
                TARGET_PAIR_TERMS, z_columns, raw_columns
            )
            if z_col is None and raw_col is None
        ]
        raise ValueError(
            "Provide either all seven standardised z-score terms or all seven raw "
            f"terms. Missing: {', '.join(missing)}"
        )

    if use_input_z:
        calibration_rows: int | str = "prestandardised"
        for (output_name, _, _, _), source in zip(TARGET_PAIR_TERMS, z_columns):
            values = pd.to_numeric(table[source], errors="raise")
            if not np.isfinite(values).all():
                raise ValueError(f"Column {source!r} contains non-finite values.")
            result[output_name] = values.astype(float)
    else:
        if calibration_column in table.columns:
            calibration_mask = _boolean_mask(table[calibration_column])
        else:
            calibration_mask = pd.Series(True, index=table.index)
        if not calibration_mask.any():
            raise ValueError("No rows were selected for feature standardisation.")
        calibration_rows = int(calibration_mask.sum())
        for (output_name, _, _, _), source in zip(TARGET_PAIR_TERMS, raw_columns):
            values = pd.to_numeric(table[source], errors="raise").astype(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Column {source!r} contains non-finite values.")
            reference = values.loc[calibration_mask]
            mean = float(reference.mean())
            std = float(reference.std(ddof=0))
            result[output_name] = 0.0 if std == 0.0 else (values - mean) / std

    score = np.zeros(len(result), dtype=float)
    for output_name, coefficient, _, _ in TARGET_PAIR_TERMS:
        score += coefficient * result[output_name].to_numpy(dtype=float)
    result["target_pair_score"] = score
    result["rank"] = result["target_pair_score"].rank(
        method="min", ascending=False
    ).astype(int)
    result["rank_percentile"] = 100.0 * result["rank"] / len(result)
    result["calibration_rows"] = calibration_rows

    order = [
        "target_pair",
        *(name for name in ("target_a", "target_b") if name in result.columns),
        "rank",
        "rank_percentile",
        "target_pair_score",
        *(term[0] for term in TARGET_PAIR_TERMS),
        "calibration_rows",
    ]
    return result.sort_values(
        ["rank", "target_pair"], kind="mergesort"
    ).loc[:, order].reset_index(drop=True)


def rank_target_pairs_file(
    input_path: str | Path,
    output_path: str | Path,
    calibration_column: str = "allowed_for_calibration",
    max_rows: int | None = None,
) -> pd.DataFrame:
    result = rank_target_pairs(
        read_csv(input_path, max_rows=max_rows),
        calibration_column=calibration_column,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def _normalise_sequences(values: Iterable[str]) -> list[str]:
    return list(
        dict.fromkeys(
            str(value).strip().upper()
            for value in values
            if str(value).strip()
        )
    )


def _split_sequences(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return _normalise_sequences(value)
    text = str(value).strip()
    if not text:
        return []
    return _normalise_sequences(
        text.replace("|", ";").replace(",", ";").split(";")
    )


def _decode_bridge_candidates(
    bundle: ModelBundle,
    hla_sequence: str,
    targets: Sequence[str],
    sigmas: Sequence[float],
    samples_per_source: int,
    max_len: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> list[dict[str, object]]:
    encoded = [_encode_condition(bundle, hla_sequence, peptide) for peptide in targets]
    bridge_mean = torch.stack([item[2] for item in encoded]).mean(dim=0)
    memory, mask = encoded[0][:2]
    routes = [(0.0, samples_per_source)]
    routes.extend((float(sigma), samples_per_source) for sigma in sigmas)

    candidates: list[dict[str, object]] = []
    for sigma, draws in routes:
        route = "bridge_direct" if sigma == 0.0 else f"bridge_sigma_{sigma:g}"
        for _ in range(draws):
            latent = (
                bridge_mean
                if sigma == 0.0
                else bridge_mean + torch.randn_like(bridge_mean) * sigma
            )
            decoded = _decode_fixed_latent(
                bundle,
                latent,
                memory,
                mask,
                num_sequences=1,
                max_len=max_len,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            for row in decoded:
                candidates.append(
                    {
                        "cdr3": bundle.tokenizer.decode(
                            row, skip_special_tokens=True
                        ).strip().upper(),
                        "generation_route": route,
                        "source_target": "bridge",
                    }
                )
    return candidates


def _candidate_pool(
    bundle: ModelBundle,
    hla_sequence: str,
    targets: Sequence[str],
    method: str,
    seed: int,
    samples_per_source: int,
    bridge_sigmas: Sequence[float],
    max_len: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    cdr3_min_len: int,
    cdr3_max_len: int,
) -> tuple[list[dict[str, object]], int]:
    candidates: list[dict[str, object]] = []
    if method == "overlap":
        for target_index, peptide in enumerate(targets):
            sequences = generate_cdr3_for_condition(
                bundle,
                hla_sequence,
                peptide,
                num_return_sequences=samples_per_source,
                strategy="nucleus",
                max_len=max_len,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed + target_index,
                cdr3_min_len=cdr3_min_len,
                cdr3_max_len=cdr3_max_len,
            )
            candidates.extend(
                {
                    "cdr3": cdr3,
                    "generation_route": "endpoint",
                    "source_target": peptide,
                }
                for cdr3 in sequences
            )

    _set_seed(seed + len(targets))
    candidates.extend(
        _decode_bridge_candidates(
            bundle,
            hla_sequence,
            targets,
            sigmas=bridge_sigmas,
            samples_per_source=samples_per_source,
            max_len=max_len,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    )

    raw_count = len(candidates)
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for candidate in candidates:
        cdr3 = str(candidate["cdr3"])
        if cdr3 in seen or not valid_cdr3(cdr3, cdr3_min_len, cdr3_max_len):
            continue
        seen.add(cdr3)
        unique.append(candidate)
    return unique, raw_count


def _generate_and_score_pool(
    bundle: ModelBundle,
    hla_sequence: str,
    targets: Sequence[str],
    off_targets: Sequence[str] | None,
    *,
    method: str,
    sample_id: str,
    seed: int,
    samples_per_source: int,
    bridge_sigmas: Sequence[float],
    max_len: int,
    pool_size: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    cdr3_min_len: int,
    cdr3_max_len: int,
    task_a_weight: float,
    task_b_weight: float,
) -> pd.DataFrame:
    target_list = _normalise_sequences(targets)
    off_target_list = _normalise_sequences(off_targets or ())
    candidates, raw_count = _candidate_pool(
        bundle,
        hla_sequence,
        target_list,
        method,
        seed,
        samples_per_source,
        bridge_sigmas,
        max_len,
        temperature,
        top_p,
        repetition_penalty,
        cdr3_min_len,
        cdr3_max_len,
    )
    valid_unique_count = len(candidates)
    candidates = candidates[:pool_size]

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        cdr3 = str(candidate["cdr3"])
        target_scores = [
            float(
                score_triplet(
                    bundle,
                    hla_sequence,
                    peptide,
                    cdr3,
                    task_a_weight=task_a_weight,
                    task_b_weight=task_b_weight,
                )["score_pair_raw"]
            )
            for peptide in target_list
        ]
        off_scores = [
            float(
                score_triplet(
                    bundle,
                    hla_sequence,
                    peptide,
                    cdr3,
                    task_a_weight=task_a_weight,
                    task_b_weight=task_b_weight,
                )["score_pair_raw"]
            )
            for peptide in off_target_list
        ]
        minimum_target_score = float(np.min(target_scores))
        multitarget_score = (
            minimum_target_score - float(np.max(off_scores))
            if off_scores
            else minimum_target_score
        )
        row: dict[str, object] = {
            "sample_id": sample_id,
            "cdr3": cdr3,
            "multitarget_score": multitarget_score,
            "minimum_target_score": minimum_target_score,
            "target_score_std": float(np.std(target_scores)),
            "target_scores": ";".join(
                f"{peptide}={score:.6f}"
                for peptide, score in zip(target_list, target_scores)
            ),
            "targets": ";".join(target_list),
            "off_targets": ";".join(off_target_list) if off_target_list else "none",
            "hla_sequence": hla_sequence,
            **candidate,
            "method": method,
            "candidate_generation": (
                "target_endpoints_and_bridge"
                if method == "overlap"
                else "shared_bridge"
            ),
            "seed": seed,
            "raw_candidates": raw_count,
            "valid_unique_candidates": valid_unique_count,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_len": max_len,
            "cdr3_min_len": cdr3_min_len,
            "cdr3_max_len": cdr3_max_len,
            "task_a_weight": task_a_weight,
            "task_b_weight": task_b_weight,
        }
        if off_scores:
            row["maximum_offtarget_score"] = float(np.max(off_scores))
            row["specificity_margin"] = multitarget_score
            row["offtarget_scores"] = ";".join(
                f"{peptide}={score:.6f}"
                for peptide, score in zip(off_target_list, off_scores)
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["multitarget_score", "cdr3"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def generate_multitarget(
    bundle: ModelBundle,
    hla_sequence: str,
    targets: Sequence[str],
    off_targets: Sequence[str] | None = None,
    *,
    method: str = "overlap",
    n: int = 10,
    pool_size: int = 100,
    seed: int = 3407,
    temperature: float = 1.0,
    top_p: float = 0.85,
    repetition_penalty: float = 1.2,
    max_len: int = 25,
    cdr3_min_len: int = 10,
    cdr3_max_len: int = 20,
    task_a_weight: float = 0.5,
    task_b_weight: float = 0.5,
    sample_id: str = "sample_0",
) -> pd.DataFrame:
    """``overlap`` samples endpoints and bridge; ``bridge`` samples only the bridge.

    Ranking rewards all targets and optionally penalises off-target support.
    """
    target_list = _normalise_sequences(targets)
    if len(target_list) < 2:
        raise ValueError("Multi-target generation requires at least two targets.")
    method = str(method).strip().lower()
    if method not in {"overlap", "bridge"}:
        raise ValueError("method must be 'overlap' or 'bridge'")
    if n < 1:
        raise ValueError("n must be positive")
    if pool_size < n:
        raise ValueError("pool_size must be at least n")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Require temperature > 0 and 0 < top_p <= 1")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if max_len < 1 or cdr3_min_len < 1 or cdr3_max_len < cdr3_min_len:
        raise ValueError("Invalid sequence-length settings")

    bridge_sigmas = (0.05, 0.10)
    endpoint_sources = len(target_list) if method == "overlap" else 0
    source_count = endpoint_sources + len(bridge_sigmas) + 1
    samples_per_source = max(1, math.ceil(1.5 * pool_size / source_count))
    result = _generate_and_score_pool(
        bundle,
        hla_sequence,
        target_list,
        off_targets,
        method=method,
        sample_id=str(sample_id),
        seed=seed,
        samples_per_source=samples_per_source,
        bridge_sigmas=bridge_sigmas,
        max_len=max_len,
        pool_size=pool_size,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        cdr3_min_len=cdr3_min_len,
        cdr3_max_len=cdr3_max_len,
        task_a_weight=task_a_weight,
        task_b_weight=task_b_weight,
    )
    if result.empty:
        return result

    scored_pool_size = len(result)
    result = result.head(n).copy()
    result["rank"] = np.arange(1, len(result) + 1)
    result["num_return_sequences"] = n
    result["candidate_pool_size"] = pool_size
    result["scored_pool_size"] = scored_pool_size

    columns = [
        "sample_id",
        "rank",
        "cdr3",
        "multitarget_score",
        "minimum_target_score",
    ]
    if "maximum_offtarget_score" in result.columns:
        columns.extend(["maximum_offtarget_score", "specificity_margin"])
    columns.extend(["target_score_std", "target_scores"])
    if "offtarget_scores" in result.columns:
        columns.append("offtarget_scores")
    columns.extend(
        [
            "targets",
            "off_targets",
            "hla_sequence",
            "generation_route",
            "source_target",
            "method",
            "candidate_generation",
            "seed",
            "num_return_sequences",
            "candidate_pool_size",
            "scored_pool_size",
            "valid_unique_candidates",
            "temperature",
            "top_p",
            "repetition_penalty",
            "max_len",
            "cdr3_min_len",
            "cdr3_max_len",
            "task_a_weight",
            "task_b_weight",
        ]
    )
    return result.loc[:, columns]


def _targets_from_group(group: pd.DataFrame) -> list[str]:
    targets_col = _first_column(group, ("targets",))
    if targets_col:
        values: list[str] = []
        for value in group[targets_col]:
            values.extend(_split_sequences(value))
        return _normalise_sequences(values)

    target_col = _first_column(group, ("target", "peptide", "antigen"))
    if target_col:
        return _normalise_sequences(group[target_col])
    target_a_col = first_existing_column(group, "target_a", required=False)
    target_b_col = first_existing_column(group, "target_b", required=False)
    if target_a_col and target_b_col:
        return _normalise_sequences(
            list(group[target_a_col]) + list(group[target_b_col])
        )
    wt_col = first_existing_column(group, "wt", required=False)
    variant_col = first_existing_column(group, "variant", required=False)
    if variant_col:
        values = list(group[variant_col])
        if wt_col:
            values = list(group[wt_col]) + values
        return _normalise_sequences(values)
    raise ValueError(
        "Input requires targets, target, target_a/target_b, or wt/variant columns."
    )


def _offtargets_from_group(group: pd.DataFrame) -> list[str]:
    column = _first_column(group, ("off_targets", "offtargets"))
    if column is None:
        return []
    values: list[str] = []
    for value in group[column]:
        values.extend(_split_sequences(value))
    return _normalise_sequences(values)


def generate_multitarget_dataframe(
    bundle: ModelBundle,
    table: pd.DataFrame,
    hla_lookup_path: str | Path | None = None,
    **options,
) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Multi-target generation input is empty.")
    sample_col = _first_column(
        table, ("sample_id", "panel_id", "pair_id", "panel", "id")
    )
    groups = table.groupby(sample_col, sort=False) if sample_col else [("sample_0", table)]
    hla_lookup = load_hla_lookup(hla_lookup_path)
    base_seed = int(options.get("seed", 3407))
    common_options = {key: value for key, value in options.items() if key != "seed"}
    outputs = []
    for offset, (sample_id, group) in enumerate(groups):
        hla_sequences = {
            resolve_hla_sequence(row, hla_lookup) for _, row in group.iterrows()
        }
        if len(hla_sequences) != 1:
            raise ValueError(f"Sample {sample_id!r} must use one shared HLA condition.")
        outputs.append(
            generate_multitarget(
                bundle,
                hla_sequences.pop(),
                _targets_from_group(group),
                _offtargets_from_group(group),
                sample_id=str(sample_id),
                seed=base_seed + offset,
                **common_options,
            )
        )
    nonempty = [table for table in outputs if not table.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def generate_multitarget_file(
    bundle: ModelBundle,
    input_path: str | Path,
    output_path: str | Path,
    max_rows: int | None = None,
    **options,
) -> pd.DataFrame:
    result = generate_multitarget_dataframe(
        bundle,
        read_csv(input_path, max_rows=max_rows),
        **options,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
