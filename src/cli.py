from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import Cognate
from .embeddings import embed_csv
from .inference import generate_csv
from .model import load_cognate
from .scoring import score_csv
from .targeting import generate_multitarget_file, rank_target_pairs_file


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Model config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face online mode.")
    parser.add_argument("--hla-lookup", default=None, help="Optional HLA pseudo-sequence CSV.")


def add_multitarget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=["overlap", "bridge"],
        default="overlap",
        help="Candidate search method: general overlap search or bridge-only search.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=10,
        help="Number of ranked CDR3β candidates to return.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=100,
        help="Maximum number of unique candidates scored before ranking.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--top-p", type=float, default=0.85, help="Nucleus probability threshold."
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.2, help="Repetition penalty."
    )
    parser.add_argument(
        "--max-len", type=int, default=25, help="Maximum decoded sequence length."
    )
    parser.add_argument(
        "--cdr3-min-len", type=int, default=10, help="Minimum accepted CDR3β length."
    )
    parser.add_argument(
        "--cdr3-max-len", type=int, default=20, help="Maximum accepted CDR3β length."
    )
    parser.add_argument(
        "--task-a-weight", type=float, default=0.5, help="Task A score weight."
    )
    parser.add_argument(
        "--task-b-weight", type=float, default=0.5, help="Task B score weight."
    )


def _load(args):
    return load_cognate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        offline=not args.online,
    )


def cmd_generate_batch(args) -> None:
    bundle = _load(args)
    result = generate_csv(
        bundle,
        args.input,
        args.out,
        max_rows=args.max_rows,
        hla_lookup_path=args.hla_lookup,
        num_return_sequences=args.num_return_sequences,
        strategy=args.strategy,
        beam_width=args.beam_width,
        max_len=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        cdr3_min_len=args.cdr3_min_len,
        cdr3_max_len=args.cdr3_max_len,
        keep_invalid=args.keep_invalid,
    )
    print(f"[COGNATE] wrote {len(result)} generated candidates to {args.out}")


def cmd_score_batch(args) -> None:
    bundle = _load(args)
    result = score_csv(
        bundle,
        args.input,
        args.out,
        max_rows=args.max_rows,
        hla_lookup_path=args.hla_lookup,
        task_a_weight=args.task_a_weight,
        task_b_weight=args.task_b_weight,
    )
    print(f"[COGNATE] wrote {len(result)} scores to {args.out}")


def cmd_embed_batch(args) -> None:
    bundle = _load(args)
    npz_path, metadata_path = embed_csv(
        bundle,
        args.input,
        args.npz,
        args.metadata,
        max_rows=args.max_rows,
        hla_lookup_path=args.hla_lookup,
        batch_size=args.batch_size,
    )
    print(f"[COGNATE] wrote embeddings to {npz_path} and metadata to {metadata_path}")


def cmd_rank_target_pairs(args) -> None:
    result = rank_target_pairs_file(
        args.input,
        args.out,
        calibration_column=args.calibration_column,
        max_rows=args.max_rows,
    )
    print("Target-pair ranking")
    print(f"  Input: {args.input}")
    print(f"  Calibration column: {args.calibration_column}")
    print(f"  Ranked pairs: {len(result)}")
    print("\nTop target pairs")
    for row in result.head(5).itertuples(index=False):
        print(
            f"  {row.rank}. {row.target_pair}: "
            f"score={row.target_pair_score:.6f}, percentile={row.rank_percentile:.2f}%"
        )
    print(f"\nOutput: {args.out}")


def _multitarget_options(args) -> dict:
    return {
        "method": args.method,
        "n": args.num_return_sequences,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_len": args.max_len,
        "cdr3_min_len": args.cdr3_min_len,
        "cdr3_max_len": args.cdr3_max_len,
        "task_a_weight": args.task_a_weight,
        "task_b_weight": args.task_b_weight,
    }


def _print_multitarget_settings(args) -> None:
    print("Generation")
    print(f"  Method: {args.method}")
    print(f"  Return sequences: {args.num_return_sequences}")
    print(f"  Candidate pool: {args.pool_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Repetition penalty: {args.repetition_penalty}")
    print(f"  Maximum length: {args.max_len}")
    print(f"  Accepted CDR3β length: {args.cdr3_min_len}-{args.cdr3_max_len}")
    print(f"  Task weights: A={args.task_a_weight}, B={args.task_b_weight}")


def cmd_generate_multitarget(args) -> None:
    model = _load_api(args)
    hla_display = args.hla or args.hla_sequence
    result = model.generate_multitarget(
        targets=args.targets,
        off_targets=args.off_targets,
        hla=args.hla,
        hla_sequence=args.hla_sequence,
        **_multitarget_options(args),
    )
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    if args.json:
        payload = {
            "input": {
                "targets": args.targets,
                "off_targets": args.off_targets,
                "hla": hla_display,
            },
            "generation": _multitarget_options(args),
            "candidates": json.loads(result.to_json(orient="records")),
        }
        print(json.dumps(payload, indent=2))
        return

    print("Input")
    print(f"  Targets: {', '.join(args.targets)}")
    print(f"  HLA: {hla_display}")
    print(
        "  Off-targets: "
        + (", ".join(args.off_targets) if args.off_targets else "not provided")
    )
    print()
    _print_multitarget_settings(args)
    print("\nGenerated multi-target CDR3β")
    if result.empty:
        print("  No valid sequence generated.")
    else:
        for row in result.itertuples(index=False):
            print(
                f"  {row.rank}. {row.cdr3}  "
                f"multi-target score={row.multitarget_score:.6f}"
            )
    if args.out:
        print(f"\nOutput: {args.out}")


def cmd_generate_multitarget_batch(args) -> None:
    bundle = _load(args)
    result = generate_multitarget_file(
        bundle,
        args.input,
        args.out,
        max_rows=args.max_rows,
        hla_lookup_path=args.hla_lookup,
        **_multitarget_options(args),
    )
    samples = result["sample_id"].nunique() if not result.empty else 0
    print("Multi-target batch generation")
    print(f"  Input: {args.input}")
    print(f"  Samples with candidates: {samples}")
    print()
    _print_multitarget_settings(args)
    print("\nOutput")
    print(f"  Candidates: {len(result)}")
    print(f"  File: {args.out}")


def _load_api(args) -> Cognate:
    return Cognate.from_pretrained(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        offline=not args.online,
        hla_lookup=args.hla_lookup,
    )


def cmd_score(args) -> None:
    model = _load_api(args)
    hla_args = {"hla": args.hla, "hla_sequence": args.hla_sequence}
    hla_display = args.hla or args.hla_sequence
    result = model.score(
        args.peptide,
        args.cdr3,
        task_a_weight=args.task_a_weight,
        task_b_weight=args.task_b_weight,
        **hla_args,
    )
    if args.json:
        payload = {
            "input": {
                "peptide": args.peptide,
                "cdr3": args.cdr3,
                "hla": hla_display,
            },
            "scoring": {
                "task_a_weight": args.task_a_weight,
                "task_b_weight": args.task_b_weight,
            },
            "scores": result,
        }
        print(json.dumps(payload, indent=2))
        return
    print("Input")
    print(f"  Peptide: {args.peptide}")
    print(f"  HLA: {hla_display}")
    print(f"  CDR3β: {args.cdr3}")
    print("\nScoring")
    print(f"  Task A weight: {args.task_a_weight}")
    print(f"  Task B weight: {args.task_b_weight}")
    print("\nLog-compatibility scores (higher is better; 0 is the upper bound)")
    print(f"  Task A (CDR3β | HLA, peptide): {result['score_task_a']:.6f}")
    print(f"  Task B (peptide | HLA, CDR3β): {result['score_task_b']:.6f}")
    print(f"  Combined bidirectional score: {result['score_pair_raw']:.6f}")
    print("  These are model scores, not binding probabilities.")


def cmd_generate(args) -> None:
    model = _load_api(args)
    hla_args = {"hla": args.hla, "hla_sequence": args.hla_sequence}
    hla_display = args.hla or args.hla_sequence
    result = model.generate(
        args.peptide,
        n=args.num_return_sequences,
        strategy=args.strategy,
        seed=args.seed,
        beam_width=args.beam_width,
        max_len=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        cdr3_min_len=args.cdr3_min_len,
        cdr3_max_len=args.cdr3_max_len,
        keep_invalid=args.keep_invalid,
        **hla_args,
    )
    generation = {
        "strategy": args.strategy,
        "seed": args.seed,
        "num_return_sequences": args.num_return_sequences,
        "max_len": args.max_len,
        "cdr3_min_len": args.cdr3_min_len,
        "cdr3_max_len": args.cdr3_max_len,
        "keep_invalid": args.keep_invalid,
    }
    if args.strategy == "beam":
        generation["beam_width"] = args.beam_width
    else:
        generation.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
            }
        )
    payload = {
        "input": {
            key: value
            for key, value in {
                "peptide": args.peptide,
                "hla": args.hla,
                "hla_sequence": args.hla_sequence,
            }.items()
            if value is not None
        },
        "generation": generation,
        "generated_cdr3": result,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print("Input")
    print(f"  Peptide: {args.peptide}")
    print(f"  HLA: {hla_display}")
    print("\nGeneration")
    print(f"  Strategy: {args.strategy}")
    if args.strategy == "beam":
        print(f"  Beam width: {args.beam_width}")
    else:
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Repetition penalty: {args.repetition_penalty}")
    print(f"  Max length: {args.max_len}")
    print(f"  Seed: {args.seed}")
    print("\nGenerated CDR3β")
    if result:
        for index, sequence in enumerate(result, start=1):
            print(f"  {index}. {sequence}")
    else:
        print("  No valid sequence generated.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Generate CDR3 candidates for one HLA-peptide condition."
    )
    add_model_args(generate)
    generate.add_argument("--peptide", required=True)
    generate.add_argument("--hla", default=None)
    generate.add_argument("--hla-sequence", default=None)
    generate.add_argument("--num-return-sequences", type=int, default=1)
    generate.add_argument("--strategy", choices=["beam", "nucleus"], default="beam")
    generate.add_argument("--beam-width", type=int, default=5)
    generate.add_argument("--max-len", type=int, default=25)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=0.85)
    generate.add_argument("--repetition-penalty", type=float, default=1.2)
    generate.add_argument("--seed", type=int, default=3407)
    generate.add_argument("--cdr3-min-len", type=int, default=10)
    generate.add_argument("--cdr3-max-len", type=int, default=20)
    generate.add_argument("--keep-invalid", action="store_true")
    generate.add_argument("--json", action="store_true", help="Print JSON output.")
    generate.set_defaults(func=cmd_generate)

    generate_multitarget = subparsers.add_parser(
        "generate-multitarget",
        help="Generate CDR3β candidates jointly prioritised across multiple targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(generate_multitarget)
    generate_multitarget.add_argument(
        "--targets", nargs="+", required=True, help="Two or more target peptides."
    )
    generate_multitarget.add_argument(
        "--off-targets",
        nargs="+",
        default=None,
        help="Optional off-target peptides; omit this argument when unused.",
    )
    generate_multitarget.add_argument("--hla", default=None)
    generate_multitarget.add_argument("--hla-sequence", default=None)
    generate_multitarget.add_argument(
        "--out", default=None, help="Optional candidate CSV output."
    )
    generate_multitarget.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    add_multitarget_args(generate_multitarget)
    generate_multitarget.set_defaults(func=cmd_generate_multitarget)

    score = subparsers.add_parser(
        "score", help="Score one HLA-peptide-CDR3 triplet."
    )
    add_model_args(score)
    score.add_argument("--peptide", required=True)
    score.add_argument("--cdr3", required=True)
    score.add_argument("--hla", default=None)
    score.add_argument("--hla-sequence", default=None)
    score.add_argument("--task-a-weight", type=float, default=0.5)
    score.add_argument("--task-b-weight", type=float, default=0.5)
    score.add_argument("--json", action="store_true", help="Print JSON output.")
    score.set_defaults(func=cmd_score)

    generate_batch = subparsers.add_parser(
        "generate-batch", help="Generate CDR3 candidates from a CSV file."
    )
    add_model_args(generate_batch)
    generate_batch.add_argument("--input", required=True)
    generate_batch.add_argument("--out", required=True)
    generate_batch.add_argument("--max-rows", type=int, default=None)
    generate_batch.add_argument("--num-return-sequences", type=int, default=1)
    generate_batch.add_argument(
        "--strategy", choices=["beam", "nucleus"], default="beam"
    )
    generate_batch.add_argument("--beam-width", type=int, default=5)
    generate_batch.add_argument("--max-len", type=int, default=25)
    generate_batch.add_argument("--temperature", type=float, default=1.0)
    generate_batch.add_argument("--top-p", type=float, default=0.85)
    generate_batch.add_argument("--repetition-penalty", type=float, default=1.2)
    generate_batch.add_argument("--seed", type=int, default=3407)
    generate_batch.add_argument("--cdr3-min-len", type=int, default=10)
    generate_batch.add_argument("--cdr3-max-len", type=int, default=20)
    generate_batch.add_argument("--keep-invalid", action="store_true")
    generate_batch.set_defaults(func=cmd_generate_batch)

    generate_multitarget_batch = subparsers.add_parser(
        "generate-multitarget-batch",
        help="Generate multi-target CDR3β candidates from a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(generate_multitarget_batch)
    generate_multitarget_batch.add_argument(
        "--input", required=True, help="Input CSV with targets and HLA columns."
    )
    generate_multitarget_batch.add_argument(
        "--out", required=True, help="Ranked candidate CSV."
    )
    generate_multitarget_batch.add_argument(
        "--max-rows", type=int, default=None, help="Optional input-row limit."
    )
    add_multitarget_args(generate_multitarget_batch)
    generate_multitarget_batch.set_defaults(func=cmd_generate_multitarget_batch)

    score_batch = subparsers.add_parser(
        "score-batch", help="Score HLA-peptide-CDR3 triplets from a CSV file."
    )
    add_model_args(score_batch)
    score_batch.add_argument("--input", required=True)
    score_batch.add_argument("--out", required=True)
    score_batch.add_argument("--max-rows", type=int, default=None)
    score_batch.add_argument("--task-a-weight", type=float, default=0.5)
    score_batch.add_argument("--task-b-weight", type=float, default=0.5)
    score_batch.set_defaults(func=cmd_score_batch)

    embed_batch = subparsers.add_parser(
        "embed-batch", help="Export representations from a CSV file."
    )
    add_model_args(embed_batch)
    embed_batch.add_argument("--input", required=True)
    embed_batch.add_argument("--npz", required=True)
    embed_batch.add_argument("--metadata", required=True)
    embed_batch.add_argument("--max-rows", type=int, default=None)
    embed_batch.add_argument("--batch-size", type=int, default=128)
    embed_batch.set_defaults(func=cmd_embed_batch)

    target_ranking = subparsers.add_parser(
        "rank-target-pairs",
        help="Rank multispecific target pairs with the fixed seven-term score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target_ranking.add_argument("--input", required=True, help="Input CSV or TSV.")
    target_ranking.add_argument("--out", required=True, help="Ranked output CSV.")
    target_ranking.add_argument(
        "--max-rows", type=int, default=None, help="Optional input-row limit."
    )
    target_ranking.add_argument(
        "--calibration-column",
        default="allowed_for_calibration",
        help="Boolean column selecting raw-feature calibration rows.",
    )
    target_ranking.set_defaults(func=cmd_rank_target_pairs)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
