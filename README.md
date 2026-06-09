# CIRCUIT-X

A two-stage framework for identifying minimal causal circuits in large language models for spatial reasoning tasks.

## Overview

CIRCUIT-X works in two stages:

1. **Stage I** — Scores each parameter group by causal importance using activation ablation across perturbed inputs.
2. **Stage II** — Optimises sigmoid-gated binary masks over the selected groups to find the smallest subnetwork that preserves accuracy.

Supports LLaMA-2-7B, Mistral-7B-Instruct, and Gemma-7B on SPARTQA, StepGame, and a real-world geography dataset.

## Requirements

- Python 3.8+
- CUDA GPU (≥16 GB VRAM recommended for 7B models in fp16; ≥40 GB total for 70B models)

```bash
pip install -r circuit_x/requirements.txt
```

## Project Structure

```
circuit_x/
├── main.py               # Entry point
├── config.py             # All hyperparameters and paths
├── models/
│   ├── backbone.py       # Model loading, parameter groups
│   └── circuit.py        # Binary mask wrapper (CircuitMask, MaskedModel)
├── stages/
│   ├── stage1.py         # Causal importance estimation
│   └── stage2.py         # Mask optimisation
├── data/
│   ├── loader.py         # SPARTQA / StepGame / Geography dataset loaders
│   └── interventions.py  # Spatial relation perturbations
├── metrics/
│   └── evaluate.py       # Acc, IR, CC, PE, AR, OS metrics
├── experiments/
│   ├── run_main.py       # In-domain evaluation
│   ├── run_cross_domain.py
│   ├── run_efficiency.py
│   ├── run_ablation.py
│   ├── run_hyperparam.py
│   ├── run_geoeval.py
│   └── run_llm_compare.py
└── baselines/
    └── run_baselines.py  # PISTAQ, SREQA, NSM, PostGIS, GeoQA baselines
```

## API Models (Optional)

For LLM comparison experiments, set environment variables before running:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

Models without a key are automatically skipped.
