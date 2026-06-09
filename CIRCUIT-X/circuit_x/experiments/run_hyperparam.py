"""Hyperparameter sensitivity analysis."""

import csv
import json
import logging
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (BACKBONE_MODELS, DATASETS, RESULTS_DIR, CHECKPOINT_DIR,
                    SEED, TAU1, TAU2, LAMBDA1, LAMBDA2, LR_STAGE2,
                    EPOCHS_STAGE2, EPSILON, BATCH_SIZE_STAGE1,
                    TAU1_SWEEP, TAU2_SWEEP, LAMBDA1_SWEEP, LAMBDA2_SWEEP)
from data.loader import load_spatial_dataset
from models.backbone import load_model_and_tokenizer, get_parameter_groups
from metrics.evaluate import compute_all_metrics, _infer_candidates
from stages.stage1 import estimate_importance
from stages.stage2 import discover_circuit

logger = logging.getLogger(__name__)

FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def _set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def _run_sweep(model, tokenizer, param_groups, train_dl, val_dl, test_dl,
               candidate_answers, base_acc, tau1, tau2, lam1, lam2,
               max_s1=None, max_s2=None, max_eval=None):
    """Run Stage I + II with given hyperparameters, return metrics dict."""
    theta_s, _ = estimate_importance(
        model, tokenizer, param_groups, train_dl,
        tau1=tau1, tau2=tau2,
        max_batches=max_s1 or BATCH_SIZE_STAGE1,
    )
    if not theta_s:
        logger.warning("Empty θ_s in sweep — no circuit formed.")
        return {}

    _, masked_model = discover_circuit(
        model, tokenizer, theta_s,
        train_dl, val_dl, candidate_answers,
        base_accuracy=base_acc,
        lambda1=lam1, lambda2=lam2,
        lr=LR_STAGE2, epochs=EPOCHS_STAGE2, epsilon=EPSILON,
        max_batches_per_epoch=max_s2,
    )

    results = compute_all_metrics(
        model_circuit=masked_model, model_full=model,
        dataloader=test_dl, tokenizer=tokenizer,
        candidate_answers=candidate_answers,
        max_batches=max_eval,
    )
    return results.get("circuit", {})


def _bar_chart(values_dict, title, ylabel, save_path, delta_os):
    """Save a grouped bar chart (CC, IR or CC, PE) with ΔOS line."""
    labels = list(values_dict.keys())
    n      = len(labels)
    x      = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Primary bars (CC and IR or PE)
    bar_keys = [k for k in list(values_dict[labels[0]].keys()) if k != "OS"]
    width    = 0.35
    offsets  = np.linspace(-(len(bar_keys)-1)*width/2,
                            (len(bar_keys)-1)*width/2, len(bar_keys))

    colors = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000"]
    for ki, (key, offset, color) in enumerate(zip(bar_keys, offsets, colors)):
        vals = [values_dict[lbl].get(key, 0) for lbl in labels]
        ax1.bar(x + offset, vals, width, label=key, color=color, alpha=0.85)

    ax1.set_xlabel(title)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in labels])
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(0, 110)

    # Secondary axis: ΔOS
    ax2 = ax1.twinx()
    dos = [delta_os.get(lbl, 0) for lbl in labels]
    ax2.plot(x, dos, "s--", color="#7030a0", label="ΔOS (pp)")
    ax2.set_ylabel("ΔOS (pp)", color="#7030a0")
    ax2.tick_params(axis="y", labelcolor="#7030a0")
    ax2.legend(loc="upper right", fontsize=8)

    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure to {save_path}")


def run_hyperparam(backbone_key="llama2_7b", dataset_key="spartqa",
                   max_stage1_batches=None, max_stage2_batches=None,
                   max_eval_batches=None):
    _set_seeds()
    os.makedirs(FIG_DIR, exist_ok=True)

    try:
        model, tokenizer, _ = load_model_and_tokenizer(backbone_key)
    except Exception as e:
        logger.error(f"Cannot load model {backbone_key}: {e}")
        return

    try:
        train_dl, val_dl, test_dl = load_spatial_dataset(dataset_key)
    except Exception as e:
        logger.error(f"Cannot load dataset {dataset_key}: {e}")
        return

    cands = []
    for b in train_dl:
        cands.extend(b["label"])
    candidate_answers = _infer_candidates(cands)
    param_groups      = get_parameter_groups(model)

    default_metrics = _run_sweep(
        model, tokenizer, param_groups, train_dl, val_dl, test_dl,
        candidate_answers, base_acc=0.86,
        tau1=TAU1, tau2=TAU2, lam1=LAMBDA1, lam2=LAMBDA2,
        max_s1=max_stage1_batches, max_s2=max_stage2_batches,
        max_eval=max_eval_batches,
    )
    default_os = default_metrics.get("OS", 0.0)

    all_rows = []

    fig3_data, fig3_dos = {}, {}
    for v in TAU1_SWEEP:
        m = _run_sweep(
            model, tokenizer, param_groups, train_dl, val_dl, test_dl,
            candidate_answers, base_acc=0.86,
            tau1=v, tau2=TAU2, lam1=LAMBDA1, lam2=LAMBDA2,
            max_s1=max_stage1_batches, max_s2=max_stage2_batches,
            max_eval=max_eval_batches,
        )
        fig3_data[v] = {"CC(%)": m.get("CC", 0), "IR(%)": m.get("IR", 0)}
        fig3_dos[v]  = m.get("OS", 0) - default_os
        all_rows.append({"figure": "3", "param": "tau1", "value": v,
                         "CC": m.get("CC"), "IR": m.get("IR"),
                         "delta_OS": fig3_dos[v]})

    _bar_chart(fig3_data, "τ₁ sweep (Figure 3)", "Percentage (%)",
               os.path.join(FIG_DIR, "figure3_tau1.png"), fig3_dos)

    fig4_data, fig4_dos = {}, {}
    for v in TAU2_SWEEP:
        m = _run_sweep(
            model, tokenizer, param_groups, train_dl, val_dl, test_dl,
            candidate_answers, base_acc=0.86,
            tau1=TAU1, tau2=v, lam1=LAMBDA1, lam2=LAMBDA2,
            max_s1=max_stage1_batches, max_s2=max_stage2_batches,
            max_eval=max_eval_batches,
        )
        fig4_data[v] = {"CC(%)": m.get("CC", 0), "IR(%)": m.get("IR", 0)}
        fig4_dos[v]  = m.get("OS", 0) - default_os
        all_rows.append({"figure": "4", "param": "tau2", "value": v,
                         "CC": m.get("CC"), "IR": m.get("IR"),
                         "delta_OS": fig4_dos[v]})

    _bar_chart(fig4_data, "τ₂ sweep (Figure 4)", "Percentage (%)",
               os.path.join(FIG_DIR, "figure4_tau2.png"), fig4_dos)

    fig5_data, fig5_dos = {}, {}
    for v in LAMBDA1_SWEEP:
        m = _run_sweep(
            model, tokenizer, param_groups, train_dl, val_dl, test_dl,
            candidate_answers, base_acc=0.86,
            tau1=TAU1, tau2=TAU2, lam1=v, lam2=LAMBDA2,
            max_s1=max_stage1_batches, max_s2=max_stage2_batches,
            max_eval=max_eval_batches,
        )
        fig5_data[v] = {"CC(%)": m.get("CC", 0), "PE(%)": m.get("PE", 0)}
        fig5_dos[v]  = m.get("OS", 0) - default_os
        all_rows.append({"figure": "5", "param": "lambda1", "value": v,
                         "CC": m.get("CC"), "PE": m.get("PE"),
                         "delta_OS": fig5_dos[v]})

    _bar_chart(fig5_data, "λ₁ sweep (Figure 5)", "Percentage (%)",
               os.path.join(FIG_DIR, "figure5_lambda1.png"), fig5_dos)

    fig6_data, fig6_dos = {}, {}
    for v in LAMBDA2_SWEEP:
        m = _run_sweep(
            model, tokenizer, param_groups, train_dl, val_dl, test_dl,
            candidate_answers, base_acc=0.86,
            tau1=TAU1, tau2=TAU2, lam1=LAMBDA1, lam2=v,
            max_s1=max_stage1_batches, max_s2=max_stage2_batches,
            max_eval=max_eval_batches,
        )
        fig6_data[v] = {"CC(%)": m.get("CC", 0), "IR(%)": m.get("IR", 0)}
        fig6_dos[v]  = m.get("OS", 0) - default_os
        all_rows.append({"figure": "6", "param": "lambda2", "value": v,
                         "CC": m.get("CC"), "IR": m.get("IR"),
                         "delta_OS": fig6_dos[v]})

    _bar_chart(fig6_data, "λ₂ sweep (Figure 6)", "Percentage (%)",
               os.path.join(FIG_DIR, "figure6_lambda2.png"), fig6_dos)

    csv_path  = os.path.join(RESULTS_DIR, "hyperparam_sweep.csv")
    json_path = os.path.join(RESULTS_DIR, "hyperparam_sweep.json")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        with open(json_path, "w") as f:
            json.dump(all_rows, f, indent=2)
        logger.info(f"Hyperparameter sweep data saved to {csv_path}")

    return all_rows
