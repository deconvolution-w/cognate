# COGNATE

This repository contains the research code accompanying **“Learning the
constraints of T-cell antigen perception with a reciprocal immune language
model.”**

COGNATE is a reciprocal immune language model that learns the constraints of
T-cell antigen perception by modeling both directions of the interaction:

- `P(CDR3β | HLA, peptide)`
- `P(peptide | HLA, CDR3β)`

The repository supports:

| Task | Python API | CLI |
|---|---|---|
| Generate CDR3β sequences | `model.generate()` | `generate`, `generate-batch` |
| Generate for multiple targets | `model.generate_multitarget()` | `generate-multitarget`, `generate-multitarget-batch` |
| Score HLA–peptide–CDR3β triplets | `model.score()` | `score`, `score-batch` |
| Rank target pairs | `model.rank_target_pairs()` | `rank-target-pairs` |
| Extract model representations | `model.embed()` | `embed-batch` |
| Train the reciprocal model | — | `scripts/train_single.sh`, `scripts/train_4gpu.sh` |

The repository provides the model architecture, losses, data processing,
training defaults, and evaluation metrics used by COGNATE.

## Installation

Python 3.10 or newer is supported. The manuscript environment used Python 3.13,
PyTorch 2.6, and CUDA 12.4.

After cloning this repository, create the environment from the repository
directory:

```bash
cd cognate

conda create -n cognate python=3.13 -y
conda activate cognate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands below from the repository root. To see the available commands:

```bash
python -m src.cli --help
```

## Model setup

### 1. Download ESM-2

COGNATE uses `facebook/esm2_t33_650M_UR50D` through Hugging Face Transformers.
Download it once into the local Hugging Face cache:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esm2_t33_650M_UR50D')"
```

Inference runs in offline mode by default. Alternatively, add `--online` to the
first CLI command to allow Transformers to download the model files.

### 2. Install the COGNATE checkpoint

The approximately 2.8 GiB checkpoint is not tracked in Git. Download it directly
from Hugging Face to the default checkpoint path:

```bash
mkdir -p checkpoints
curl -L \
  https://huggingface.co/hanw98/cognate/resolve/main/checkpoint.pth \
  -o checkpoints/checkpoint.pth
```

Its expected SHA-256 checksum and byte size are recorded in
`checkpoints/README.md`. The inference configuration and checkpoint paths are
resolved automatically. Use `--config` or `--checkpoint` to override them.

### Configuration layout

- `configs/model.yaml`: canonical model architecture shared by every mode.
- `configs/inference.yaml`: inference settings inheriting the canonical model.
- `configs/train_single.yaml`: single-GPU/CPU training configuration.
- `configs/train_4gpu.yaml`: four-GPU distributed training configuration.

## Quick start

### Python API

```python
from src import Cognate

model = Cognate.from_pretrained(device="cuda:0")

# Generate a CDR3β sequence.
cdr3_sequences = model.generate(
    peptide="GLCTLVAML",
    hla="HLA-A*02:01",
    n=1,
    strategy="beam",
)

# Score an HLA–peptide–CDR3β triplet.
score = model.score(
    peptide="GLCTLVAML",
    cdr3="CASSLGQGAEAFF",
    hla="HLA-A*02:01",
)
pair_score = score["score_pair_raw"]  # Higher (closer to zero) is better.

# Extract the full triplet representation.
features = model.embed(
    peptide="GLCTLVAML",
    cdr3="CASSLGQGAEAFF",
    hla="HLA-A*02:01",
)
embedding = features["full_1472"]
```

### Command line

```bash
python -m src.cli generate \
  --peptide GLCTLVAML \
  --hla 'HLA-A*02:01' \
  --device cuda:0
```

Device selection is automatic when `--device` is omitted. Single-item commands
also support `--json` for machine-readable output.

## Generation

The quick-start command above uses deterministic beam search, the default
strategy for primary top-1 generation. Its default beam width is `5`.

### Diverse nucleus sampling

Use nucleus sampling to produce a more diverse candidate set:

```bash
python -m src.cli generate \
  --peptide GLCTLVAML \
  --hla 'HLA-A*02:01' \
  --strategy nucleus \
  --num-return-sequences 10
```

All generation interfaces use seed `3407` by default. Override `--seed` only
when a different reproducible run is required.

### Batch generation

The input CSV must contain `peptide` and either `hla` or `hla_sequence`. See
`data/examples/generation_inputs.csv` for a complete example.

```bash
python -m src.cli generate-batch \
  --input data/examples/generation_inputs.csv \
  --out outputs/generated.csv
```

The output records each input condition, generated sequence, selected strategy,
and the decoding parameters used by that strategy. Add `--max-rows N` to process
only the first `N` input rows.

## Multi-target generation

Multi-target generation searches for CDR3β candidates supported across two or
more peptide targets under one HLA condition. Optionally, provide `off_targets`
to penalize support for unwanted peptides.

```python
candidates = model.generate_multitarget(
    targets=["GLATLVAML", "GLCALVAML"],
    hla="HLA-A*02:01",
    off_targets=["GILGFVFTL", "NLVPMVATV"],
    method="overlap",
    n=10,
    pool_size=100,
)
```

The method returns a pandas DataFrame ordered from the strongest to the weakest
candidate.

| Method | Recommended use | Candidate pool |
|---|---|---|
| `overlap` (default) | Targets are distinct, dispersed, or their geometry is unknown | Every target endpoint plus the shared bridge neighbourhood |
| `bridge` | Targets are known to be closely related variants | Shared bridge neighbourhood only |

### Single target set

```bash
python -m src.cli generate-multitarget \
  --targets GLATLVAML GLCALVAML \
  --off-targets GILGFVFTL NLVPMVATV \
  --hla 'HLA-A*02:01' \
  --out outputs/multitarget.csv
```

### Multiple target sets

Use one CSV row per target set. The `targets` column contains semicolon-separated
peptides; the optional `off_targets` column uses the same format. See
`data/examples/multitarget_inputs.csv` for an example with multiple target sets.

```bash
python -m src.cli generate-multitarget-batch \
  --input data/examples/multitarget_inputs.csv \
  --out outputs/multitarget_batch.csv
```

COGNATE removes invalid and duplicate sequences, scores each candidate under all
targets, and ranks candidates by their minimum target score. When off-targets are
provided, the maximum off-target score is subtracted from the ranking score.

## Triplet scoring

COGNATE scores the compatibility of an HLA–peptide–CDR3β triplet. Scores are
typically negative; higher values (closer to zero) indicate stronger model
compatibility. They are ranking scores, not binding probabilities.

```bash
python -m src.cli score \
  --peptide GLCTLVAML \
  --cdr3 CASSLGQGAEAFF \
  --hla 'HLA-A*02:01'
```

For batch scoring, provide a CSV containing `peptide`, `cdr3`, and either `hla`
or `hla_sequence`. See `data/examples/pairs.csv`.

```bash
python -m src.cli score-batch \
  --input data/examples/pairs.csv \
  --out outputs/scores.csv
```

The two reciprocal scoring directions are weighted equally by default. Use
`--task-a-weight` and `--task-b-weight` only when custom weights are required.

## Target-pair ranking

Rank a table containing all seven raw or standardized target-pair features:

```bash
python -m src.cli rank-target-pairs \
  --input target_pairs.csv \
  --out outputs/ranked_target_pairs.csv
```

## Representation extraction

Use `model.embed()` to extract peptide, TCR, pMHC, or joint TCR–pMHC
representations:

| Representation | Required input | Composition | Returned key | Dimension |
|---|---|---|---|---:|
| Peptide | `peptide` | `peptide_condition_640` + `peptide_prior_96` | `peptide_block_736` | 736 |
| TCR | `cdr3` | `tcr_condition_640` + `tcr_prior_96` | `tcr_block_736` | 736 |
| pMHC | `hla` + `peptide` | `pmhc_condition_640` + `pmhc_prior_96` | `pmhc_block_736` | 736 |
| Joint TCR–pMHC | `hla` + `peptide` + `cdr3` | `pmhc_block_736` + `tcr_block_736` | `full_1472` | 1,472 |

Here, `+` denotes vector concatenation. ESM-2 fusion is enabled for all
representation blocks.

### Batch export

```bash
python -m src.cli embed-batch \
  --input data/examples/pairs.csv \
  --npz outputs/embeddings.npz \
  --metadata outputs/embeddings.csv
```

The `.npz` file stores the available 640- and 96-dimensional components as
separate arrays, together with the corresponding 736-dimensional blocks and,
for complete triplets, `full_1472`. The metadata CSV maps input rows to arrays
and records representation names, dimensions, and batch size. Use
`--max-rows N` to limit the export.

## Generation evaluation

Evaluate the released checkpoint on the complete held-out test split with the
same beam-search and metric definitions used for the reported results:

```bash
python scripts/evaluate_generation_metrics.py \
  --input data/training/test.csv \
  --config configs/inference.yaml \
  --checkpoint checkpoints/checkpoint.pth \
  --device cuda:0 \
  --out outputs/test_evaluation \
  --batch-size 32 \
  --beam-width 5 \
  --max-len 25
```

The released checkpoint produces approximately:

| Metric | COGNATE |
|---|---:|
| CDR3β Recovery ↑ | 0.559 |
| Character-level BLEU ↑ | 0.286 |
| BLOSUM62 ↑ | 18.34 |
| k-mer Jensen–Shannon distance ↓ | 0.567 |

## Training

Training, validation, and held-out test splits are included in `data/training/`.
Training uses one canonical implementation: four-process Accelerate DDP with
BF16 mixed precision. The configured batch size is per GPU.

Run the formal four-GPU training from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_4gpu.sh
```

The launcher above is equivalent to:

```bash
mkdir -p outputs/train_4gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=. \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  --main_process_port 0 \
  --rdzv_backend static \
  --module src.training \
  --config configs/train_4gpu.yaml \
  --output outputs/train_4gpu \
  2>&1 | tee outputs/train_4gpu/training.log
```

The canonical model configuration is `configs/model.yaml`. The four-GPU
training settings are in `configs/train_4gpu.yaml`, which inherits that model.
The training configuration retains the manuscript reciprocal task weights, KL
schedule, BLOSUM62 and physicochemical losses, HLA omission, denoising,
gradient clipping, cosine schedule, and early stopping.
Logs, history CSV/JSON, TensorBoard events, and the best checkpoint are written
to `outputs/train_4gpu/`.

To continue an interrupted run from its last saved best epoch, pass the same
output directory as the third argument:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_4gpu.sh \
  configs/train_4gpu.yaml outputs/train_4gpu outputs/train_4gpu
```

Weights-only checkpoints do not include optimizer state. Resuming restores the
epoch, history, best validation loss, and learning-rate position, but
reinitializes optimizer and random-number-generator state.

For adaptive single-device training, run:

```bash
bash scripts/train_single.sh
```

The single-device training settings are in `configs/train_single.yaml`. Both
training configurations inherit `configs/model.yaml`, so the model
architecture is identical. Their losses and optimisation settings are also
kept aligned. The single-device launcher uses one CUDA GPU when CUDA is
available and falls back to CPU otherwise.
Device selection is controlled by `runtime.device`, and its GPU/CPU batch sizes
are controlled by `data.gpu_batch_size` and `data.cpu_batch_size`.

To select a particular GPU, expose only that device:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/train_single.sh
```

To force CPU even when a GPU is available, change the configuration to:

```yaml
runtime:
  device: cpu
```

The batch size is also configured in YAML:

```yaml
data:
  batch_size: 4
  gpu_batch_size: 4
  cpu_batch_size: 1
```

To follow training or inspect TensorBoard:

```bash
tail -f outputs/train_4gpu/training.log
tensorboard --logdir outputs/train_4gpu/tensorboard --port 6006
```

## Citation

If you use COGNATE in your research, please cite the accompanying manuscript:

> *Learning the constraints of T-cell antigen perception with a reciprocal
> immune language model.*

The complete bibliographic record can be added when the manuscript record is
publicly available.

## License

This source code is licensed under the MIT License. See `LICENSE` for details.
