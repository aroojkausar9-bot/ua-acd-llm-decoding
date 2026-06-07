# Evaluating and Improving Factual Consistency in Long-Form LLM Generation

**Large Language Models**
  Research-Oriented Project

**Team**
- Arooj Kausar
- Areeba Hassan 
- Syeda Kisaa Fatima 

---

## Overview

This repository implements UA-ACD (Uncertainty-Aware Adaptive Constraint Decoding), a system that dynamically adjusts NLI verification stringency based on per-claim uncertainty during long-form generation. It is evaluated against three baselines on the FactScore biography benchmark.

**Five pipelines:**
1. Vanilla LLM — greedy decoding, no retrieval
2. Standard RAG — retrieval-augmented generation, static prompt
3. Static Constraint — RAG with fixed NLI threshold across all claims
4. UA-ACD — adaptive NLI thresholds conditioned on claim-level uncertainty
5. UA-ACD-Dec-Mod — UA-ACD with entropy-conditioned decoding temperature

---

## Repository Structure

```
project/
  README.md
  requirements.txt
  environment.yml
  .env.example                  # no secrets required; see file for details
  configs/
    default.yaml                # all hyperparameters
  src/
    data/dataset.py             # FactScore + Wikipedia data loading
    models/
      retriever.py              # Hybrid BM25 + FAISS retrieval
      verifier.py               # NLI verifier, claim segmenter, adaptive controller
      generator.py              # All five generation pipelines
    evaluation/metrics.py       # Evaluation loop and metric aggregation
    utils/config.py             # Config loader
    service/__init__.py         # Stub — Track A has no HTTP service component
  scripts/
    run_pipeline.py             # Main entry point (runs evaluation end-to-end)
    eval.py                     # Recompute metrics/figures from saved CSVs
    train.py                    # Stub — UA-ACD is inference-only (no training phase)
  tests/
    test_verifier.py
    test_retriever.py
  figures/                      # Pre-generated result figures
  artifacts/
    logs/                       # Saved CSVs from Kaggle run
  reports/
      Report.pdf             # Compiled report PDF
      main.tex                  # LaTeX source
      bibliography.bib
      acl.sty                   # ACL style file (included for clean-env compilation)
      acl_natbib.bst            # ACL bibliography style
      figures/                  # Figures for the report
  notebooks_reference.ipynb     # Exploration notebook (secondary; not the graded pipeline)
```

> **Note on `scripts/train.py`:** UA-ACD is an inference-only method. The generator (Phi-2) and NLI verifier are used as frozen pretrained models with no fine-tuning. `train.py` is included as a required structural stub and exits with an explanatory message.

> **Note on `src/service/`:** This is a Track A research project. No HTTP API or service layer is implemented. The package stub is included to satisfy the required folder structure.

---

## Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ua-acd
python -m spacy download en_core_web_sm
```

### Option B: pip

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **CPU-only run:** set `load_in_4bit: false` in `configs/default.yaml` before running. The generator model (Phi-2, 2.7B) will load in fp32 and run on CPU. Expect roughly 3-5 minutes per topic on a modern CPU.

---

## Reproducing Results

### Full evaluation (all five methods, 50 topics)

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

Results are saved to `artifacts/logs/ua_acd_all_results.csv` and `artifacts/logs/summary_table.csv`.

### Evaluate a subset of methods

```bash
python scripts/run_pipeline.py --config configs/default.yaml --methods uaacd static
```

Available method keys: `vanilla`, `rag`, `static`, `uaacd`, `decmod`.

### Recompute metrics and figures from pre-saved results

```bash
python scripts/eval.py --results artifacts/logs/ua_acd_all_results.csv --out-dir figures
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Compiling the Report

The LaTeX source is fully self-contained in `reports/`. All required style files (`acl.sty`, `acl_natbib.bst`) are included so it compiles in a clean environment without any additional installation.

```bash
cd reports
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled PDF is also provided as `reports/Report.pdf`.

---

## Configuration

All parameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.generator_model_id` | `microsoft/phi-2` | HuggingFace model ID for the generator |
| `model.load_in_4bit` | `true` | 4-bit quantisation (GPU only) |
| `evaluation.n_eval_samples` | `50` | Number of topics to evaluate |
| `uncertainty.low_threshold` | `0.30` | Boundary between low and mid uncertainty tiers |
| `uncertainty.high_threshold` | `0.70` | Boundary between mid and high uncertainty tiers |
| `nli_thresholds.low/mid/high` | `0.60/0.75/0.85` | Per-tier NLI support thresholds |
| `evaluation.seed` | `42` | Random seed for all sampling operations |

---

## Pre-generated Results

The `artifacts/logs/` directory contains CSVs from the full Kaggle run (T4 GPU, 50 topics each):

| File | Description |
|------|-------------|
| `ua_acd_all_results.csv` | Per-topic results for all five methods |
| `summary_table.csv` | Aggregated mean/std metrics per method |
| `ablation1_uncertainty_components.csv` | Ablation 1: entropy vs consistency weighting |
| `ablation2_thresholds.csv` | Ablation 2: NLI threshold sensitivity |
| `ablation3_granularity.csv` | Ablation 3: sentence vs atomic claim granularity |

Figures in `figures/` are pre-generated from the same run.

---

## Hardware and Reproducibility

- **Original run:** Kaggle T4 GPU (16 GB VRAM), CUDA 11.8, Python 3.10
- **Seed:** 42 (fixed for dataset sampling, model generation, and numpy)
- **Dataset:** FactScore test split (`shmsw25/FActScoring`), Wikipedia 20220301.en
- **Model:** `microsoft/phi-2` loaded in 4-bit NF4 quantisation via bitsandbytes
- **NLI model:** `cross-encoder/nli-deberta-v3-small` on CPU
