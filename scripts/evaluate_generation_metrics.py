#!/usr/bin/env python3
"""Evaluate CDR3 recovery, BLEU, BLOSUM62, and k-mer JSD."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.biochemistry import BLOSUM62_RAW
from src.inference import generate_cdr3_for_conditions
from src.model import load_cognate


def _column(table: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in table.columns:
            return name
    raise ValueError(f"Missing required column; accepted names: {', '.join(names)}")


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _aligned_stats(reference: str, prediction: str, gap_penalty: int = -4):
    """Needleman-Wunsch BLOSUM score and identity of the selected alignment."""
    rows, columns = len(reference) + 1, len(prediction) + 1
    scores = np.zeros((rows, columns), dtype=np.int32)
    trace = np.zeros((rows, columns), dtype=np.int8)
    scores[:, 0] = np.arange(rows) * gap_penalty
    scores[0, :] = np.arange(columns) * gap_penalty
    trace[1:, 0] = 1
    trace[0, 1:] = 2

    for row in range(1, rows):
        for column in range(1, columns):
            substitution = BLOSUM62_RAW[reference[row - 1]][prediction[column - 1]]
            choices = (
                scores[row - 1, column - 1] + substitution,
                scores[row - 1, column] + gap_penalty,
                scores[row, column - 1] + gap_penalty,
            )
            direction = int(np.argmax(choices))
            scores[row, column] = choices[direction]
            trace[row, column] = direction

    row, column = len(reference), len(prediction)
    matches = 0
    alignment_length = 0
    while row or column:
        direction = int(trace[row, column])
        if row and column and direction == 0:
            matches += reference[row - 1] == prediction[column - 1]
            row -= 1
            column -= 1
        elif row and (column == 0 or direction == 1):
            row -= 1
        else:
            column -= 1
        alignment_length += 1
    identity = matches / max(1, alignment_length)
    return float(scores[-1, -1]), float(identity)


def _ngrams(sequence: str, order: int) -> Counter:
    return Counter(
        sequence[index : index + order]
        for index in range(max(0, len(sequence) - order + 1))
    )


def _corpus_bleu(references: list[str], predictions: list[str], max_order: int = 4):
    clipped = np.zeros(max_order, dtype=np.float64)
    totals = np.zeros(max_order, dtype=np.float64)
    reference_length = 0
    prediction_length = 0
    for reference, prediction in zip(references, predictions):
        reference_length += len(reference)
        prediction_length += len(prediction)
        for order in range(1, max_order + 1):
            ref_counts = _ngrams(reference, order)
            pred_counts = _ngrams(prediction, order)
            clipped[order - 1] += sum(
                min(count, ref_counts[gram]) for gram, count in pred_counts.items()
            )
            totals[order - 1] += sum(pred_counts.values())
    precisions = np.divide(
        clipped,
        totals,
        out=np.zeros_like(clipped),
        where=totals > 0,
    )
    if not np.all(precisions > 0) or prediction_length == 0:
        return 0.0
    brevity_penalty = (
        1.0
        if prediction_length > reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return float(brevity_penalty * math.exp(np.log(precisions).mean()))


def _sentence_bleu_method1(reference: str, prediction: str, max_order: int = 4):
    """NLTK-compatible sentence BLEU-4 with SmoothingFunction.method1."""
    precisions = []
    for order in range(1, max_order + 1):
        reference_counts = _ngrams(reference, order)
        prediction_counts = _ngrams(prediction, order)
        clipped = sum(
            min(count, reference_counts[gram])
            for gram, count in prediction_counts.items()
        )
        total = sum(prediction_counts.values())
        precisions.append(clipped / total if clipped else 0.1 / max(1, total))
    if not prediction:
        return 0.0
    brevity_penalty = (
        1.0
        if len(prediction) > len(reference)
        else math.exp(1.0 - len(reference) / len(prediction))
    )
    return float(brevity_penalty * math.exp(np.log(precisions).mean()))


def _kmer_jsd(references: list[str], predictions: list[str], k: int) -> float:
    reference_counts = Counter()
    prediction_counts = Counter()
    for sequence in references:
        reference_counts.update(_ngrams(sequence, k))
    for sequence in predictions:
        prediction_counts.update(_ngrams(sequence, k))
    vocabulary = sorted(reference_counts.keys() | prediction_counts.keys())
    reference = np.array([reference_counts[value] for value in vocabulary], np.float64)
    prediction = np.array([prediction_counts[value] for value in vocabulary], np.float64)
    reference /= reference.sum()
    prediction /= prediction.sum()
    midpoint = 0.5 * (reference + prediction)

    def divergence(values):
        valid = values > 0
        return float(np.sum(values[valid] * np.log2(values[valid] / midpoint[valid])))

    return math.sqrt(0.5 * divergence(reference) + 0.5 * divergence(prediction))


def _kmer_jsd_natural_log(
    references: list[str], predictions: list[str], k: int, boundary_tokens: bool
) -> float:
    reference_counts = Counter()
    prediction_counts = Counter()
    for sequence in references:
        value = f"^{sequence}$" if boundary_tokens else sequence
        reference_counts.update(_ngrams(value, k))
    for sequence in predictions:
        value = f"^{sequence}$" if boundary_tokens else sequence
        prediction_counts.update(_ngrams(value, k))
    vocabulary = sorted(reference_counts.keys() | prediction_counts.keys())
    reference = np.array([reference_counts[value] for value in vocabulary], np.float64)
    prediction = np.array([prediction_counts[value] for value in vocabulary], np.float64)
    reference /= reference.sum()
    prediction /= prediction.sum()
    midpoint = 0.5 * (reference + prediction)

    def divergence(values):
        valid = values > 0
        return float(np.sum(values[valid] * np.log(values[valid] / midpoint[valid])))

    return math.sqrt(0.5 * divergence(reference) + 0.5 * divergence(prediction))


def calculate_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    aligned = [_aligned_stats(ref, pred) for ref, pred in zip(references, predictions)]
    positional_blosum = [
        sum(BLOSUM62_RAW[left][right] for left, right in zip(ref, pred))
        - 4 * abs(len(ref) - len(pred))
        for ref, pred in zip(references, predictions)
    ]
    positional = [
        sum(a == b for a, b in zip(ref, pred)) / max(1, max(len(ref), len(pred)))
        for ref, pred in zip(references, predictions)
    ]
    edit_recovery = [
        1.0 - _levenshtein(ref, pred) / max(1, max(len(ref), len(pred)))
        for ref, pred in zip(references, predictions)
    ]
    kmer_distances = {
        f"kmer_{k}_jensen_shannon_distance": _kmer_jsd(
            references, predictions, k
        )
        for k in range(1, 5)
    }
    return {
        "cdr3_recovery_aligned": float(np.mean([value[1] for value in aligned])),
        "cdr3_recovery_positional": float(np.mean(positional)),
        "cdr3_recovery_edit": float(np.mean(edit_recovery)),
        "exact_sequence_recovery": float(
            np.mean([ref == pred for ref, pred in zip(references, predictions)])
        ),
        "character_sentence_bleu4_method1": float(
            np.mean(
                [
                    _sentence_bleu_method1(ref, pred)
                    for ref, pred in zip(references, predictions)
                ]
            )
        ),
        "character_corpus_bleu4": _corpus_bleu(references, predictions),
        "blosum62_positional_gap_minus4": float(np.mean(positional_blosum)),
        "blosum62_global_alignment": float(np.mean([value[0] for value in aligned])),
        **kmer_distances,
        "kmer_jensen_shannon_distance_mean_1_to_4": float(
            np.mean(list(kmer_distances.values()))
        ),
        "kmer_3_jensen_shannon_distance_natural_log_boundaries": (
            _kmer_jsd_natural_log(references, predictions, 3, True)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/test.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=25)
    args = parser.parse_args()

    table = pd.read_csv(args.input)
    if args.max_rows:
        table = table.head(args.max_rows).copy()
    peptide_column = _column(table, ("peptide", "epitope", "antigen"))
    hla_column = _column(table, ("hla_sequence", "HLA_sequence", "MHC", "hla"))
    cdr3_column = _column(table, ("cdr3", "CDR3", "tcr", "TCR"))
    peptides = table[peptide_column].astype(str).str.strip().str.upper().tolist()
    hla_sequences = table[hla_column].astype(str).str.strip().str.upper().tolist()
    references = table[cdr3_column].astype(str).str.strip().str.upper().tolist()

    bundle = load_cognate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    predictions: list[str] = []
    for start in range(0, len(table), args.batch_size):
        stop = min(start + args.batch_size, len(table))
        generated = generate_cdr3_for_conditions(
            bundle,
            hla_sequences[start:stop],
            peptides[start:stop],
            num_return_sequences=1,
            strategy="beam",
            beam_width=args.beam_width,
            max_len=args.max_len,
            batch_size=args.batch_size,
            keep_invalid=True,
        )
        predictions.extend(rows[0] if rows else "" for rows in generated)
        if stop == len(table) or (start // args.batch_size + 1) % 10 == 0:
            print(f"[COGNATE] generated {stop}/{len(table)}", flush=True)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.DataFrame(
        {
            "peptide": peptides,
            "hla_sequence": hla_sequences,
            "reference_cdr3": references,
            "predicted_cdr3": predictions,
        }
    )
    prediction_path = output / "predictions.csv"
    prediction_table.to_csv(prediction_path, index=False)
    metrics = {
        "rows": len(references),
        "device": args.device,
        "strategy": "beam",
        "beam_width": args.beam_width,
        "max_len": args.max_len,
        **calculate_metrics(references, predictions),
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"[COGNATE] predictions: {prediction_path}")
    print(f"[COGNATE] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
