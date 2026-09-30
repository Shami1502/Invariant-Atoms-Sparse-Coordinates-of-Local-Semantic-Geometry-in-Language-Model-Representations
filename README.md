# Invariant Atoms: Sparse Coordinates of Local Semantic Geometry in Language Model Representations

It evaluates the Global and Local invariant-atom frames on the held-out data used in the paper, for Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.

## Files

```
evaluate.py                 evaluation script
requirements.txt
data/eval.json              held-out evaluation split (1,400 anchors, 16,581 SP and 13,997 SC variants)
checkpoints/llama/          global.pt, local.pt
checkpoints/qwen/           global.pt, local.pt
```

## Setup

```bash
pip install -r requirements.txt
```

The script uses a CUDA GPU if available, then Apple Silicon (MPS), then CPU. Each model needs about 16 GB of memory in float16.

Llama 3.1 is a gated model. Before the first run, request access on its Hugging Face page (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and log in with an access token:

```bash
huggingface-cli login
```

Alternatively, set the token as an environment variable with `export HF_TOKEN=<your token>`. Qwen2.5 does not require access approval. Both models are downloaded automatically on the first run (about 16 GB each); set `HF_HOME` to choose where they are stored.

## Usage

Llama:

```bash
python evaluate.py \
    --model meta-llama/Llama-3.1-8B-Instruct --layer 16 --hidden-dim 4096 \
    --global-ckpt checkpoints/llama/global.pt \
    --local-ckpt checkpoints/llama/local.pt
```

Qwen:

```bash
python evaluate.py \
    --model Qwen/Qwen2.5-7B-Instruct --layer 20 --hidden-dim 3584 \
    --global-ckpt checkpoints/qwen/global.pt \
    --local-ckpt checkpoints/qwen/local.pt
```

The same command works for any other causal language model: set `--model`, `--layer` and `--hidden-dim`, and provide matching checkpoints.

| Argument | Description |
|---|---|
| `--model` | Hugging Face model id or local path |
| `--layer` | primary layer, as an index into `hidden_states` (0 = embeddings) |
| `--hidden-dim` | expected hidden size; the run stops on a mismatch |
| `--global-ckpt`, `--local-ckpt` | frame checkpoints (either can be omitted) |
| `--data` | evaluation data (default `data/eval.json`) |
| `--topk` | sparsity budget k (default 32) |
| `--dtype` | `float16` (default), `bfloat16` or `float32` |
| `--batch-size`, `--max-length` | extraction batch size (64) and token limit (64) |
| `--save-features`, `--features` | save extracted features and reuse them to skip extraction |
| `--breakdown` | also print S/T per SC type and nuisance energy per SP style |
| `--output` | write full results (with standard deviations and counts) to a JSON file |
| `--device` | `cuda`, `mps` or `cpu` |



## Protocol

Per-dimension standardization statistics (μ, σ) are computed from the evaluation data provided here.

`data/eval.json` contains the odd-indexed SP and SC variants of every anchor, which form the held-out split; the even-indexed variants were used for training. For the Local frame, A(x) is computed from the anchor and used for all projections in its group.

## Metrics

For an anchor z, a held-out SC variant gives Δz_sem = z_sc − z and a held-out SP variant gives Δz_nui = z_sp − z. The sparse code is α = TopK(W Δz_sem + b, k). Metrics are averaged over held-out SC samples.

| Metric | Definition |
|---|---|
| E_rec | ‖Δz_sem − Aα‖ / ‖Δz_sem‖ |
| Cos | cos(Δz_sem, Aα) |
| S/T | ‖AᵀΔz_sem‖² divided by the mean ‖AᵀΔz_nui‖² over the held-out SP variants of the same anchor |
| D_coord | mean over the first min(5, M) held-out SP variants of ‖Aᵀ(z_sc − z) − Aᵀ(z_sc − z_sp)‖² / ‖Aᵀ(z_sc − z)‖² |
| EffNum | (Σ_j \|α_j\|)² / Σ_j \|α_j\|² |
