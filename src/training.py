from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object, set_seed as accelerate_set_seed
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .core import AminoAcidTokenizer, CognateCVAE
from .core.batching import ReciprocalCollator
from .core.biochemistry import get_aa_props_tensor, get_blosum62_tensor
from .core.losses import ReciprocalLoss
from .config import load_config
from .inference import generate_cdr3_for_conditions
from .model import ModelBundle


class SequenceDataset(Dataset):
    def __init__(self, path: str | Path, max_rows: int | None = None) -> None:
        table = pd.read_csv(path, nrows=max_rows)
        aliases = {
            "cdr3": ("cdr3", "CDR3", "tcr", "TCR"),
            "peptide": ("peptide", "epitope", "antigen"),
            "hla": ("hla_sequence", "HLA_sequence", "MHC", "HLA", "hla"),
        }
        columns: dict[str, str] = {}
        for canonical, choices in aliases.items():
            columns[canonical] = next(
                (name for name in choices if name in table.columns), ""
            )
            if not columns[canonical]:
                raise ValueError(f"{path} is missing a {canonical} column")
        clean = table[[columns["cdr3"], columns["peptide"], columns["hla"]]].dropna()
        clean.columns = ["cdr3", "peptide", "hla"]
        self.rows = clean.astype(str).to_dict("records")
        if not self.rows:
            raise ValueError(f"No usable rows in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _compute_direction_loss(
    model: CognateCVAE,
    criterion: ReciprocalLoss,
    batch: dict[str, Any],
) -> dict[str, Any]:
    from .core.utils import causal_mask

    tgt_in = batch["tgt"][:, :-1]
    target = batch["tgt"][:, 1:]
    pad_in = batch["tgt_pad_mask"][:, :-1]
    output = model(
        src=batch["src"],
        tgt=tgt_in,
        src_mask=batch["src_mask"],
        tgt_pad_mask=pad_in,
        tgt_causal_mask=causal_mask(tgt_in.size(1), tgt_in.device),
        segment_ids=batch["segment_ids"],
        esm_hla_src=batch.get("esm_hla_src"),
        esm_hla_mask=batch.get("esm_hla_mask"),
        esm_pep_src=batch.get("esm_pep_src"),
        esm_pep_mask=batch.get("esm_pep_mask"),
        hla_len=batch.get("hla_len"),
        pep_len=batch.get("pep_len"),
    )
    return criterion(output, target)


def _resolve_config_path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


METRIC_NAMES = (
    "total_loss",
    "task_a_loss",
    "task_a_reconstruction",
    "task_a_kl",
    "task_a_blosum",
    "task_a_physicochemical",
    "task_b_loss",
    "task_b_reconstruction",
    "task_b_kl",
    "task_b_blosum",
    "task_b_physicochemical",
)


def _batch_metrics(
    task_a: dict[str, torch.Tensor],
    task_b: dict[str, torch.Tensor],
    lambda_a: float,
    lambda_b: float,
) -> dict[str, torch.Tensor]:
    return {
        "total_loss": lambda_a * task_a["loss"] + lambda_b * task_b["loss"],
        "task_a_loss": task_a["loss"],
        "task_a_reconstruction": task_a["reconstruction"],
        "task_a_kl": task_a["kl"],
        "task_a_blosum": task_a["blosum"],
        "task_a_physicochemical": task_a["physicochemical"],
        "task_b_loss": task_b["loss"],
        "task_b_reconstruction": task_b["reconstruction"],
        "task_b_kl": task_b["kl"],
        "task_b_blosum": task_b["blosum"],
        "task_b_physicochemical": task_b["physicochemical"],
    }


def _accumulate(
    sums: dict[str, float], metrics: dict[str, torch.Tensor], count: int
) -> None:
    for name in METRIC_NAMES:
        sums[name] += float(metrics[name].detach()) * count


def _distributed_means(
    accelerator: Accelerator, sums: dict[str, float], count: int
) -> dict[str, float]:
    packed = torch.tensor(
        [*(sums[name] for name in METRIC_NAMES), float(count)],
        dtype=torch.float64,
        device=accelerator.device,
    )
    packed = accelerator.reduce(packed, reduction="sum")
    denominator = max(1.0, float(packed[-1]))
    return {
        name: float(packed[index]) / denominator
        for index, name in enumerate(METRIC_NAMES)
    }


def _write_history(output: Path, history: list[dict[str, Any]]) -> None:
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    (output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


def _evaluate_generation_metrics(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: AminoAcidTokenizer,
    protein_tokenizer: Any,
    config: dict[str, Any],
    config_file: Path,
    output: Path,
    rows: list[dict[str, str]],
    options: dict[str, Any],
) -> dict[str, float]:
    local_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if index % accelerator.num_processes == accelerator.process_index
    ]
    references = [row["cdr3"].strip().upper() for _, row in local_rows]
    peptides = [row["peptide"].strip().upper() for _, row in local_rows]
    hla_sequences = [row["hla"].strip().upper() for _, row in local_rows]
    bundle = ModelBundle(
        model=accelerator.unwrap_model(model),
        tokenizer=tokenizer,
        config=config,
        device=str(accelerator.device),
        config_path=config_file,
        checkpoint_path=output / "best_model.pth",
        offline=True,
        _esm_tokenizer=protein_tokenizer,
    )
    with accelerator.autocast():
        generated = generate_cdr3_for_conditions(
            bundle,
            hla_sequences,
            peptides,
            num_return_sequences=1,
            strategy="beam",
            beam_width=int(options.get("beam_width", 5)),
            max_len=int(options.get("max_len", 25)),
            batch_size=int(options.get("batch_size", 16)),
            keep_invalid=True,
        )
    local_results = [
        (index, reference, predictions[0] if predictions else "")
        for (index, _), reference, predictions in zip(
            local_rows, references, generated
        )
    ]
    gathered = gather_object(local_results)
    if not accelerator.is_main_process:
        return {}
    gathered.sort(key=lambda value: value[0])
    all_references = [value[1] for value in gathered]
    all_predictions = [value[2] for value in gathered]
    # Validation and held-out evaluation share the same metric definitions.
    from scripts.evaluate_generation_metrics import calculate_metrics

    raw = calculate_metrics(all_references, all_predictions)
    return {
        "cdr3_recovery": raw["cdr3_recovery_edit"],
        "character_bleu": raw["character_sentence_bleu4_method1"],
        "blosum62": raw["blosum62_positional_gap_minus4"],
        "kmer_js_distance": raw[
            "kmer_jensen_shannon_distance_mean_1_to_4"
        ],
    }


def train(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    device_mode: str | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any] | None:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    base = config_file.parent
    train_cfg = config["training"]
    accumulation_steps = max(
        1, int(train_cfg.get("gradient_accumulation_steps", 1))
    )
    if device_mode not in (None, "auto", "cpu", "cuda"):
        raise ValueError(f"Unsupported device mode: {device_mode}")
    single_device = device_mode is not None
    cuda_available = torch.cuda.is_available()
    if device_mode == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    use_cpu = single_device and (
        device_mode == "cpu" or (device_mode == "auto" and not cuda_available)
    )
    requested_precision = str(train_cfg.get("mixed_precision", "bf16"))
    if use_cpu:
        mixed_precision = "no"
    elif (
        single_device
        and requested_precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        mixed_precision = "fp16"
    else:
        mixed_precision = requested_precision
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        cpu=use_cpu,
        gradient_accumulation_steps=accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    if single_device and accelerator.num_processes != 1:
        raise RuntimeError(
            "The adaptive single-device entry point must run as one process."
        )
    seed = int(config.get("project", {}).get("seed", 3407))
    accelerate_set_seed(seed, device_specific=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if output_dir is None:
        output = _resolve_config_path(
            base, train_cfg.get("output_dir", "outputs/training")
        )
    else:
        output = Path(output_dir).expanduser().resolve()
    resume_dir = (
        Path(resume_from).expanduser().resolve() if resume_from is not None else None
    )
    resume_history: list[dict[str, Any]] = []
    resume_best: dict[str, Any] | None = None
    resume_checkpoint: Path | None = None
    if resume_dir is not None:
        if resume_dir != output:
            raise ValueError("Resume directory must match the output directory")
        history_path = resume_dir / "history.json"
        best_epoch_path = resume_dir / "best_epoch.json"
        resume_checkpoint = resume_dir / "best_model.pth"
        for required in (history_path, best_epoch_path, resume_checkpoint):
            if not required.is_file():
                raise FileNotFoundError(f"Resume artifact not found: {required}")
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        loaded_best = json.loads(best_epoch_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_history, list) or not loaded_history:
            raise ValueError(f"Resume history is empty or invalid: {history_path}")
        if not isinstance(loaded_best, dict):
            raise ValueError(f"Best-epoch metadata is invalid: {best_epoch_path}")
        resume_history = loaded_history
        resume_best = loaded_best
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    writer = (
        SummaryWriter(log_dir=str(output / "tensorboard"))
        if accelerator.is_main_process
        else None
    )
    if writer is not None:
        writer.add_text(
            "Config", f"```json\n{json.dumps(config, indent=2)}\n```", 0
        )

    tokenizer = AminoAcidTokenizer()
    model = CognateCVAE(tokenizer.vocab_size, config)
    if resume_checkpoint is not None:
        checkpoint_state = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        incompatible = model.load_state_dict(checkpoint_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Resume checkpoint is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del checkpoint_state
    protein_tokenizer = None
    if config["model"].get("esm", {}).get("enabled", False):
        from transformers import EsmTokenizer

        protein_tokenizer = EsmTokenizer.from_pretrained(
            config["model"]["esm"]["model_name"], local_files_only=True
        )

    data_cfg = config["data"]
    train_data = SequenceDataset(
        _resolve_config_path(base, data_cfg["train_file"]),
        max_rows=data_cfg.get("train_max_rows"),
    )
    val_data = SequenceDataset(
        _resolve_config_path(base, data_cfg["val_file"]),
        max_rows=data_cfg.get("val_max_rows"),
    )
    train_collator = ReciprocalCollator(
        tokenizer,
        protein_tokenizer=protein_tokenizer,
        training=True,
        mask_probability=float(train_cfg.get("denoising", {}).get("mask_prob", 0.1)),
        hla_include_probability=float(
            train_cfg.get("mhc_random_include_prob", 0.5)
        ),
    )
    val_collator = ReciprocalCollator(
        tokenizer, protein_tokenizer=protein_tokenizer, training=False
    )
    if single_device:
        device_prefix = "gpu" if accelerator.device.type == "cuda" else "cpu"
        batch_size = int(
            data_cfg.get(
                f"{device_prefix}_batch_size", data_cfg.get("batch_size", 32)
            )
        )
        workers = int(
            data_cfg.get(
                f"{device_prefix}_num_workers", data_cfg.get("num_workers", 0)
            )
        )
    else:
        batch_size = int(data_cfg.get("batch_size", 32))
        workers = int(data_cfg.get("num_workers", 0))
    pin_memory = accelerator.device.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("num_epochs", 1))
    start_epoch = int(resume_history[-1]["epoch"]) if resume_history else 0
    if start_epoch >= epochs:
        raise ValueError(
            f"Resume history already reached epoch {start_epoch}; "
            f"configured total is {epochs}"
        )
    # The scheduler uses unsharded steps because Accelerate advances it per rank.
    unsharded_batches_per_epoch = len(train_loader)
    batches_per_rank = math.ceil(
        unsharded_batches_per_epoch / accelerator.num_processes
    )
    optimizer_steps_per_epoch = math.ceil(batches_per_rank / accumulation_steps)
    optimizer_total_steps = max(1, epochs * optimizer_steps_per_epoch)
    schedule_total_steps = max(1, epochs * unsharded_batches_per_epoch)
    scheduler_last_epoch = -1
    if start_epoch:
        # Restore scheduler position for a weights-only checkpoint.
        scheduler_steps = (
            start_epoch * optimizer_steps_per_epoch * accelerator.num_processes
        )
        for parameter_group in optimizer.param_groups:
            parameter_group.setdefault("initial_lr", parameter_group["lr"])
        scheduler_last_epoch = scheduler_steps - 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=schedule_total_steps,
        last_epoch=scheduler_last_epoch,
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    aa_ids = [tokenizer.token2id[value] for value in tokenizer.amino_acids]
    criterion = ReciprocalLoss(
        config=config.get("loss", {}),
        blosum_matrix=get_blosum62_tensor(tokenizer.vocab, accelerator.device),
        property_matrix=get_aa_props_tensor(tokenizer.vocab, accelerator.device),
        aa_token_ids=aa_ids,
    ).to(accelerator.device)
    lambda_a = float(train_cfg.get("lambda_task_a", 2.0))
    lambda_b = float(train_cfg.get("lambda_task_b", 1.0))
    beta_max = float(train_cfg.get("beta_kl", 0.01))
    anneal_steps = int(train_cfg.get("kl_annealing", {}).get("total_steps", 1200))
    early_stopping = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_stopping.get("enabled", True))
    minimum_epochs = int(early_stopping.get("min_epochs", 20))
    patience = int(early_stopping.get("patience", 10))
    min_delta = float(early_stopping.get("min_delta", 1e-5))
    global_step = int(resume_history[-1]["global_step"]) if resume_history else 0
    best_val = (
        float(resume_best["val_total_loss"])
        if resume_best is not None
        else float("inf")
    )
    best_epoch = int(resume_best["epoch"]) if resume_best is not None else 0
    epochs_without_improvement = max(0, start_epoch - best_epoch)
    history = list(resume_history)
    gradient_check_done = False
    generation_cfg = dict(train_cfg.get("generation_metrics", {}))
    if single_device:
        generation_batch_key = (
            "gpu_batch_size"
            if accelerator.device.type == "cuda"
            else "cpu_batch_size"
        )
        generation_cfg["batch_size"] = generation_cfg.get(
            generation_batch_key, generation_cfg.get("batch_size", 16)
        )
    generation_enabled = bool(generation_cfg.get("enabled", False))
    generation_every = max(1, int(generation_cfg.get("every_epochs", 1)))
    generation_rows = min(
        len(val_data), int(generation_cfg.get("max_rows", 256))
    )
    generation_panel = val_data.rows[:generation_rows]

    accelerator.print(f"[Info] Running on {accelerator.device}")
    accelerator.print(f"[Info] Distributed: {accelerator.num_processes} GPUs")
    accelerator.print(f"[Info] TensorBoard logging to {output / 'tensorboard'}")
    accelerator.print(f"[Info] Vocab Size: {tokenizer.vocab_size}")
    accelerator.print(
        f"[Info] Trainable Parameters: "
        f"{sum(parameter.numel() for parameter in trainable):,}"
    )
    accelerator.print(
        f"[Info] Batch per GPU: {batch_size} | "
        f"Global batch: {batch_size * accelerator.num_processes * accumulation_steps}"
    )
    accelerator.print(
        f"[Info] Train rows: {len(train_data)} | Valid rows: {len(val_data)} | "
        f"Steps per epoch: {len(train_loader)}"
    )
    if resume_checkpoint is not None:
        accelerator.print(
            f"[Info] Resuming from epoch {start_epoch + 1}/{epochs}: "
            f"{resume_checkpoint}"
        )
        accelerator.print(
            "[Info] Resume checkpoint has model weights only; optimizer and "
            "RNG states were reinitialized."
        )
    accelerator.print("[Info] Start Training Loop...")

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_sums: dict[str, float] = defaultdict(float)
        epoch_count = 0
        epoch_lr_sum = 0.0
        epoch_beta_sum = 0.0
        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
        )
        for reciprocal in train_iterator:
            criterion.update_step(
                global_step + 1,
                schedule_total_steps,
                beta_max,
                anneal_steps,
            )
            task_a_batch = _move_batch_to_device(
                reciprocal["task_a"], accelerator.device
            )
            task_b_batch = _move_batch_to_device(
                reciprocal["task_b"], accelerator.device
            )
            with accelerator.accumulate(model):
                task_a = _compute_direction_loss(model, criterion, task_a_batch)
                # Separate reciprocal passes to avoid overlapping DDP hooks.
                accelerator.backward(lambda_a * task_a["loss"])
                task_b = _compute_direction_loss(model, criterion, task_b_batch)
                accelerator.backward(lambda_b * task_b["loss"])
                metrics = _batch_metrics(task_a, task_b, lambda_a, lambda_b)
                if accelerator.sync_gradients:
                    if not gradient_check_done:
                        unwrapped = accelerator.unwrap_model(model)
                        missing = [
                            name
                            for name, parameter in unwrapped.named_parameters()
                            if parameter.requires_grad and parameter.grad is None
                        ]
                        if missing:
                            raise RuntimeError(
                                "Trainable parameters without gradients: "
                                + ", ".join(missing)
                            )
                        gradient_check_done = True
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Distributed loss reporting averages batches, not rows.
            _accumulate(epoch_sums, metrics, 1)
            epoch_count += 1
            epoch_lr_sum += float(optimizer.param_groups[0]["lr"])
            epoch_beta_sum += float(criterion.beta_kl)
            if accelerator.sync_gradients:
                global_step += 1
            if accelerator.is_main_process:
                train_iterator.set_postfix(
                    {
                        "loss": f"{float(metrics['total_loss'].detach()):.4f}",
                        "rec_c": f"{float(task_a['reconstruction']):.4f}",
                        "blo_c": f"{float(task_a['blosum']):.4f}",
                        "phy_c": f"{float(task_a['physicochemical']):.4f}",
                        "kl_c": f"{float(task_a['kl']):.4f}",
                        "rec_e": f"{float(task_b['reconstruction']):.4f}",
                        "blo_e": f"{float(task_b['blosum']):.4f}",
                        "phy_e": f"{float(task_b['physicochemical']):.4f}",
                        "kl_e": f"{float(task_b['kl']):.4f}",
                    }
                )
        train_metrics = _distributed_means(accelerator, epoch_sums, epoch_count)
        model.eval()
        validation_sums: dict[str, float] = defaultdict(float)
        validation_count = 0
        validation_iterator = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1} [Valid]",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for reciprocal in validation_iterator:
                task_a_batch = _move_batch_to_device(
                    reciprocal["task_a"], accelerator.device
                )
                task_b_batch = _move_batch_to_device(
                    reciprocal["task_b"], accelerator.device
                )
                task_a = _compute_direction_loss(model, criterion, task_a_batch)
                task_b = _compute_direction_loss(model, criterion, task_b_batch)
                metrics = _batch_metrics(task_a, task_b, lambda_a, lambda_b)
                _accumulate(validation_sums, metrics, 1)
                validation_count += 1
                if accelerator.is_main_process:
                    validation_iterator.set_postfix(
                        {"loss": f"{float(metrics['total_loss'].detach()):.4f}"}
                    )
        val_metrics = _distributed_means(
            accelerator, validation_sums, validation_count
        )

        generation_metrics: dict[str, float] = {}
        if generation_enabled and (epoch + 1) % generation_every == 0:
            generation_metrics = _evaluate_generation_metrics(
                accelerator,
                model,
                tokenizer,
                protein_tokenizer,
                config,
                config_file,
                output,
                generation_panel,
                generation_cfg,
            )

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "learning_rate": epoch_lr_sum / max(1, epoch_count),
            "kl_beta": epoch_beta_sum / max(1, epoch_count),
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
            **generation_metrics,
        }
        if history:
            previous = history[-1]
            for name in METRIC_NAMES:
                record[f"delta_train_{name}"] = (
                    record[f"train_{name}"] - previous[f"train_{name}"]
                )
                record[f"delta_val_{name}"] = (
                    record[f"val_{name}"] - previous[f"val_{name}"]
                )
        history.append(record)
        accelerator.print(
            f"Epoch {epoch + 1:03d} | "
            f"LR: {record['learning_rate']:.1e} | "
            f"Beta: {record['kl_beta']:.1e}\n"
            f"  [Train] loss={train_metrics['total_loss']:.4f}, "
            f"rec_c={train_metrics['task_a_reconstruction']:.4f}, "
            f"blo_c={train_metrics['task_a_blosum']:.4f}, "
            f"phy_c={train_metrics['task_a_physicochemical']:.4f}, "
            f"kl_c={train_metrics['task_a_kl']:.4f}, "
            f"rec_e={train_metrics['task_b_reconstruction']:.4f}, "
            f"blo_e={train_metrics['task_b_blosum']:.4f}, "
            f"phy_e={train_metrics['task_b_physicochemical']:.4f}, "
            f"kl_e={train_metrics['task_b_kl']:.4f}\n"
            f"  [Valid] loss={val_metrics['total_loss']:.4f}, "
            f"rec_c={val_metrics['task_a_reconstruction']:.4f}, "
            f"blo_c={val_metrics['task_a_blosum']:.4f}, "
            f"phy_c={val_metrics['task_a_physicochemical']:.4f}, "
            f"kl_c={val_metrics['task_a_kl']:.4f}, "
            f"rec_e={val_metrics['task_b_reconstruction']:.4f}, "
            f"blo_e={val_metrics['task_b_blosum']:.4f}, "
            f"phy_e={val_metrics['task_b_physicochemical']:.4f}, "
            f"kl_e={val_metrics['task_b_kl']:.4f}",
            flush=True,
        )
        if accelerator.is_main_process and generation_metrics:
            print(
                "  [Evaluation Metrics] "
                f"Recovery={generation_metrics['cdr3_recovery']:.4f} "
                f"BLEU={generation_metrics['character_bleu']:.4f} "
                f"BLOSUM62={generation_metrics['blosum62']:.2f} "
                f"kmer_JSD={generation_metrics['kmer_js_distance']:.4f}",
                flush=True,
            )

        if writer is not None:
            writer.add_scalar("Train/Total_Loss", train_metrics["total_loss"], epoch)
            writer.add_scalar("Train/CDR3_Loss", train_metrics["task_a_loss"], epoch)
            writer.add_scalar("Train/Epi_Loss", train_metrics["task_b_loss"], epoch)
            writer.add_scalar(
                "Train/Recon_Loss",
                0.5
                * (
                    train_metrics["task_a_reconstruction"]
                    + train_metrics["task_b_reconstruction"]
                ),
                epoch,
            )
            writer.add_scalar(
                "Train/KL_Raw",
                0.5 * (train_metrics["task_a_kl"] + train_metrics["task_b_kl"]),
                epoch,
            )
            writer.add_scalar(
                "Train/Blosum_Loss",
                0.5
                * (train_metrics["task_a_blosum"] + train_metrics["task_b_blosum"]),
                epoch,
            )
            writer.add_scalar(
                "Train/Phys_Loss",
                0.5
                * (
                    train_metrics["task_a_physicochemical"]
                    + train_metrics["task_b_physicochemical"]
                ),
                epoch,
            )
            writer.add_scalar("Train/rec_c", train_metrics["task_a_reconstruction"], epoch)
            writer.add_scalar("Train/blo_c", train_metrics["task_a_blosum"], epoch)
            writer.add_scalar("Train/phy_c", train_metrics["task_a_physicochemical"], epoch)
            writer.add_scalar("Train/kl_c", train_metrics["task_a_kl"], epoch)
            writer.add_scalar("Train/rec_e", train_metrics["task_b_reconstruction"], epoch)
            writer.add_scalar("Train/blo_e", train_metrics["task_b_blosum"], epoch)
            writer.add_scalar("Train/phy_e", train_metrics["task_b_physicochemical"], epoch)
            writer.add_scalar("Train/kl_e", train_metrics["task_b_kl"], epoch)
            writer.add_scalar("Valid/Total_Loss", val_metrics["total_loss"], epoch)
            writer.add_scalar("Valid/CDR3_Loss", val_metrics["task_a_loss"], epoch)
            writer.add_scalar("Valid/Epi_Loss", val_metrics["task_b_loss"], epoch)
            writer.add_scalar(
                "Valid/Recon_Loss",
                0.5
                * (
                    val_metrics["task_a_reconstruction"]
                    + val_metrics["task_b_reconstruction"]
                ),
                epoch,
            )
            writer.add_scalar(
                "Valid/KL_Raw",
                0.5 * (val_metrics["task_a_kl"] + val_metrics["task_b_kl"]),
                epoch,
            )
            writer.add_scalar(
                "Valid/Blosum_Loss",
                0.5 * (val_metrics["task_a_blosum"] + val_metrics["task_b_blosum"]),
                epoch,
            )
            writer.add_scalar(
                "Valid/Phys_Loss",
                0.5
                * (
                    val_metrics["task_a_physicochemical"]
                    + val_metrics["task_b_physicochemical"]
                ),
                epoch,
            )
            writer.add_scalar("Valid/rec_c", val_metrics["task_a_reconstruction"], epoch)
            writer.add_scalar("Valid/blo_c", val_metrics["task_a_blosum"], epoch)
            writer.add_scalar("Valid/phy_c", val_metrics["task_a_physicochemical"], epoch)
            writer.add_scalar("Valid/kl_c", val_metrics["task_a_kl"], epoch)
            writer.add_scalar("Valid/rec_e", val_metrics["task_b_reconstruction"], epoch)
            writer.add_scalar("Valid/blo_e", val_metrics["task_b_blosum"], epoch)
            writer.add_scalar("Valid/phy_e", val_metrics["task_b_physicochemical"], epoch)
            writer.add_scalar("Valid/kl_e", val_metrics["task_b_kl"], epoch)
            writer.add_scalar("Params/LR", record["learning_rate"], epoch)
            writer.add_scalar("Params/Beta_KL", record["kl_beta"], epoch)
            for name, value in generation_metrics.items():
                writer.add_scalar(f"Generation/{name}", value, epoch)
            writer.flush()

        previous_best = best_val
        improved = val_metrics["total_loss"] < best_val - min_delta
        if improved:
            best_val = val_metrics["total_loss"]
            epochs_without_improvement = 0
        else:
            if early_enabled and epoch + 1 >= minimum_epochs:
                epochs_without_improvement += 1
                accelerator.print(
                    f"  [!] EarlyStopping counter: {epochs_without_improvement} "
                    f"out of {patience} (monitor=total_loss, "
                    f"min_epochs={minimum_epochs})"
                )
        if improved:
            accelerator.print(
                f"  [*] New Best (total_loss): {best_val:.6f} "
                f"(was {previous_best:.6f})"
            )
        accelerator.wait_for_everyone()
        accelerator.print("-" * 80)
        if accelerator.is_main_process:
            _write_history(output, history)
            if improved:
                accelerator.save(
                    accelerator.unwrap_model(model).state_dict(),
                    output / "best_model.pth",
                )
                (output / "best_epoch.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
        accelerator.wait_for_everyone()

        if (
            early_enabled
            and epoch + 1 >= minimum_epochs
            and epochs_without_improvement >= patience
        ):
            accelerator.print(
                f"[Info] Early stopping triggered after {epoch + 1} epochs."
            )
            break

    local_peak = (
        torch.cuda.max_memory_allocated(accelerator.device) / 1024**3
        if accelerator.device.type == "cuda"
        else 0.0
    )
    peak_tensor = torch.tensor(local_peak, device=accelerator.device)
    maximum_peak = float(accelerator.reduce(peak_tensor, reduction="max"))
    checkpoint_path = output / "best_model.pth"
    result = {
        "status": "ok",
        "device": str(accelerator.device),
        "distributed_processes": accelerator.num_processes,
        "mixed_precision": accelerator.mixed_precision,
        "per_gpu_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "global_effective_batch_size": (
            batch_size * accelerator.num_processes * accumulation_steps
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "best_validation_loss": best_val,
        "epochs_completed": len(history),
        "peak_cuda_memory_allocated_gib": maximum_peak,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(output / "history.csv"),
    }
    is_main_process = accelerator.is_main_process
    if is_main_process:
        (output / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        if writer is not None:
            writer.close()
        print("[Info] Training Completed.", flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return result if is_main_process else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    train(args.config, args.output, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
