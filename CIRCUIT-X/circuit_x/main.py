"""CIRCUIT-X entry point — runs all experiments.
"""

import argparse
import json
import logging
import os
import platform
import random
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (BACKBONE_MODELS, DATASETS, RESULTS_DIR, CHECKPOINT_DIR, SEED,
                    NUM_GPUS, VRAM_PER_GPU_GB, TOTAL_VRAM_GB,
                    QUANTIZE_7B, QUANTIZE_70B, CAN_RUN_70B)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(RESULTS_DIR, "circuit_x.log")
                            if os.path.isdir(RESULTS_DIR)
                            else "circuit_x.log"),
    ],
)
logger = logging.getLogger("main")

os.makedirs(RESULTS_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def log_environment():
    gpus = [torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]
    env = {
        "platform":        platform.platform(),
        "python":          sys.version,
        "seed":            SEED,
        "cuda":            torch.cuda.is_available(),
        "num_gpus":        NUM_GPUS,
        "gpus":            gpus,
        "vram_per_gpu_gb": round(VRAM_PER_GPU_GB, 1),
        "total_vram_gb":   round(TOTAL_VRAM_GB, 1),
        "quantize_7b":     QUANTIZE_7B,
        "quantize_70b":    QUANTIZE_70B,
        "can_run_70b":     CAN_RUN_70B,
    }
    env["torch"] = torch.__version__
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import datasets as hf_ds
        env["datasets"] = hf_ds.__version__
    except Exception:
        pass

    path = os.path.join(RESULTS_DIR, "environment.json")
    with open(path, "w") as f:
        json.dump(env, f, indent=2)
    logger.info(f"Environment logged to {path}")
    return env


def parse_args():
    p = argparse.ArgumentParser(description="CIRCUIT-X experiment runner")
    p.add_argument(
        "--experiment",
        default="all",
        choices=["all", "table4", "table5", "table6", "table7",
                 "table8", "table9", "table10", "figures"],
        help="Which experiment to run",
    )
    p.add_argument(
        "--backbone",
        default="all",
        choices=["all", "llama2", "mistral", "gemma"],
        help="Which backbone to use",
    )
    p.add_argument(
        "--dataset",
        default="both",
        choices=["spartqa", "stepgame", "both"],
        help="Which dataset(s) to use",
    )
    p.add_argument(
        "--max-stage1-batches", type=int, default=None,
        help="Limit Stage I batches (for quick testing)",
    )
    p.add_argument(
        "--max-stage2-batches", type=int, default=None,
        help="Limit Stage II batches per epoch (for quick testing)",
    )
    p.add_argument(
        "--max-eval-batches", type=int, default=None,
        help="Limit evaluation batches (e.g. 3 for a quick smoke test, None=full)",
    )
    p.add_argument(
        "--quantize", action="store_true",
        help="Load models with 4-bit quantisation (for <16 GB GPUs)",
    )
    return p.parse_args()


def _resolve_backbone_keys(backbone_arg: str):
    key_map = {
        "llama2":  "llama2_7b",
        "mistral": "mistral_7b",
        "gemma":   "gemma_7b",
        "all":     None,
    }
    v = key_map.get(backbone_arg)
    if v is None:
        return list(BACKBONE_MODELS.keys())
    return [v]


def _resolve_dataset_keys(dataset_arg: str):
    if dataset_arg == "both":
        return list(DATASETS.keys())
    return [dataset_arg]


def _print_table(rows, title="Results"):
    if not rows:
        print(f"\n[{title}] No results to display.\n")
        return
    keys = list(rows[0].keys())
    col_w = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows))
             for k in keys}
    sep = "+-" + "-+-".join("-" * col_w[k] for k in keys) + "-+"
    hdr = "| " + " | ".join(k.ljust(col_w[k]) for k in keys) + " |"
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        row = "| " + " | ".join(str(r.get(k, "")).ljust(col_w[k]) for k in keys) + " |"
        print(row)
    print(sep)


def main():
    args = parse_args()
    env  = log_environment()
    logger.info(f"GPUs: {env['num_gpus']}× {env['gpus']} | "
                f"{env['vram_per_gpu_gb']} GB each | {env['total_vram_gb']} GB total | "
                f"quantize_7B={env['quantize_7b']} | can_run_70B={env['can_run_70b']}")
    logger.info(f"Experiment: {args.experiment} | Backbone: {args.backbone} | "
                f"Dataset: {args.dataset}")

    backbone_keys = _resolve_backbone_keys(args.backbone)
    dataset_keys  = _resolve_dataset_keys(args.dataset)
    s1b  = args.max_stage1_batches
    s2b  = args.max_stage2_batches
    evb  = args.max_eval_batches
    if evb:
        logger.info(f"Evaluation limited to {evb} batches (smoke-test mode)")

    exp = args.experiment
    all_results = {}

    if exp in ("all", "table4"):
        logger.info("\n" + "="*60 + "\nRunning Table 4 (Main Results)\n" + "="*60)
        from experiments.run_main import run_table4
        rows = run_table4(backbone_keys=backbone_keys, dataset_keys=dataset_keys,
                          max_stage1_batches=s1b, max_stage2_batches=s2b,
                          max_eval_batches=evb)
        _print_table(rows, "Table 4: Main Results")
        all_results["table4"] = rows

    if exp in ("all", "table5"):
        logger.info("\n" + "="*60 + "\nRunning Table 5 (Cross-Domain)\n" + "="*60)
        from experiments.run_cross_domain import run_cross_domain
        rows = run_cross_domain(backbone_keys=backbone_keys,
                                max_stage1_batches=s1b, max_stage2_batches=s2b,
                                max_eval_batches=evb)
        _print_table(rows, "Table 5: Cross-Domain Transfer")
        all_results["table5"] = rows

    if exp in ("all", "table6"):
        logger.info("\n" + "="*60 + "\nRunning Table 6 (Efficiency)\n" + "="*60)
        from experiments.run_efficiency import run_efficiency
        rows = run_efficiency(backbone_keys=backbone_keys, dataset_keys=dataset_keys,
                              max_stage1_batches=s1b, max_stage2_batches=s2b)
        _print_table(rows, "Table 6: Computational Efficiency")
        all_results["table6"] = rows

    if exp in ("all", "table7", "table8"):
        logger.info("\n" + "="*60 + "\nRunning Tables 7 & 8 (Ablation)\n" + "="*60)
        from experiments.run_ablation import run_ablation
        rows7, rows8 = run_ablation(
            backbone_key="llama2_7b", dataset_keys=dataset_keys,
            max_stage1_batches=s1b, max_stage2_batches=s2b,
            max_eval_batches=evb,
        )
        _print_table(rows7, "Table 7: Ablation Study")
        _print_table(rows8, "Table 8: Causal Stability Analysis")
        all_results["table7"] = rows7
        all_results["table8"] = rows8

    if exp in ("all", "figures"):
        logger.info("\n" + "="*60 + "\nRunning Figures 3–6 (Hyperparam Sensitivity)\n" + "="*60)
        from experiments.run_hyperparam import run_hyperparam
        rows = run_hyperparam(
            backbone_key="llama2_7b", dataset_key="spartqa",
            max_stage1_batches=s1b, max_stage2_batches=s2b,
            max_eval_batches=evb,
        )
        _print_table(rows or [], "Figures 3–6: Hyperparameter Sensitivity")
        all_results["figures"] = rows or []

    if exp in ("all", "table9"):
        logger.info("\n" + "="*60 + "\nRunning Table 9 (Geoinformatics)\n" + "="*60)
        from experiments.run_geoeval import run_geoeval
        rows = run_geoeval(backbone_keys=backbone_keys,
                           max_stage1_batches=s1b, max_stage2_batches=s2b,
                           max_eval_batches=evb)
        _print_table(rows, "Table 9: Real-World Geoinformatics")
        all_results["table9"] = rows

    if exp in ("all", "table10"):
        logger.info("\n" + "="*60 + "\nRunning Table 10 (LLM Comparison)\n" + "="*60)
        from experiments.run_llm_compare import run_llm_compare
        rows = run_llm_compare(dataset_keys=dataset_keys,
                               max_stage1_batches=s1b, max_stage2_batches=s2b)
        _print_table(rows, "Table 10: Comparison with Recent LLMs")
        all_results["table10"] = rows

    logger.info("\n" + "="*60 + "\nAll selected experiments complete.\n" + "="*60)
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Full summary saved to {summary_path}")

    print("\n[Summary] Results saved in:")
    print(f"  {RESULTS_DIR}/table*.csv  — per-table CSVs")
    print(f"  {RESULTS_DIR}/figures/    — Figures 3–6 PNG plots")
    print(f"  {RESULTS_DIR}/summary.json — combined results")
    print(f"  {RESULTS_DIR}/skipped_experiments.log — skipped runs")


if __name__ == "__main__":
    main()
