"""Ablation study and causal stability analysis."""

import csv
import json
import logging
import os
import random

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (BACKBONE_MODELS, DATASETS, RESULTS_DIR, CHECKPOINT_DIR,
                    SEED, TAU1, TAU2, LAMBDA1, LAMBDA2, LR_STAGE2,
                    EPOCHS_STAGE2, EPSILON, BATCH_SIZE_STAGE1,
                    ABLATION_VARIANTS)
from data.loader import load_spatial_dataset
from models.backbone import load_model_and_tokenizer, get_parameter_groups
from metrics.evaluate import (compute_all_metrics, _infer_candidates,
                               parameter_efficiency)
from stages.stage1 import estimate_importance
from stages.stage2 import discover_circuit

logger = logging.getLogger(__name__)


def _set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def _run_variant(variant, model, tokenizer, param_groups,
                 train_dl, val_dl, test_dl,
                 candidate_answers, base_acc,
                 max_s1=None, max_s2=None, max_eval=None):
    """Run a single ablation variant, return (masked_model_or_None, metrics_dict)."""
    try:
        if variant == "base":
            results = compute_all_metrics(
                model_circuit=None, model_full=model,
                dataloader=test_dl, tokenizer=tokenizer,
                candidate_answers=candidate_answers,
                compute_stability=True,
                max_batches=max_eval,
            )
            return None, results["full_model"]

        elif variant == "stage1_no_causal_filter":
            theta_s, _ = estimate_importance(
                model, tokenizer, param_groups, train_dl,
                tau1=TAU1, tau2=1.0,   # τ2=1.0 keeps everything that passes τ1
                max_batches=max_s1 or BATCH_SIZE_STAGE1,
            )

        elif variant == "stage1_full":
            theta_s, _ = estimate_importance(
                model, tokenizer, param_groups, train_dl,
                tau1=TAU1, tau2=TAU2,
                max_batches=max_s1 or BATCH_SIZE_STAGE1,
            )

        elif variant in ("stage2_no_sparsity", "stage2_no_intervention", "full_circuit_x"):
            theta_s, _ = estimate_importance(
                model, tokenizer, param_groups, train_dl,
                tau1=TAU1, tau2=TAU2,
                max_batches=max_s1 or BATCH_SIZE_STAGE1,
            )

        else:
            raise ValueError(f"Unknown variant: {variant}")

        if not theta_s:
            logger.warning(f"  Variant {variant}: empty θ_s — skipping Stage II.")
            return None, {}

        l1 = 0.0    if variant == "stage2_no_sparsity"     else LAMBDA1
        l2 = 0.0    if variant == "stage2_no_intervention" else LAMBDA2

        ckpt = os.path.join(CHECKPOINT_DIR, "ablation", variant)
        circuit_mask, masked_model = discover_circuit(
            model, tokenizer, theta_s,
            train_dl, val_dl, candidate_answers,
            base_accuracy=base_acc,
            lambda1=l1, lambda2=l2,
            lr=LR_STAGE2, epochs=EPOCHS_STAGE2, epsilon=EPSILON,
            max_batches_per_epoch=max_s2,
            checkpoint_path=ckpt,
        )

        results = compute_all_metrics(
            model_circuit=masked_model, model_full=model,
            dataloader=test_dl, tokenizer=tokenizer,
            candidate_answers=candidate_answers,
            compute_stability=True,
            max_batches=max_eval,
        )
        return masked_model, results.get("circuit", {})

    except Exception as e:
        logger.error(f"  Variant {variant} failed: {e}")
        return None, {}


def _compute_drop(model, masked_model, tokenizer, src_dl, tgt_dl,
                  candidate_answers, max_eval=None):
    """Accuracy drop when transferring: (acc on src train domain) - (acc on tgt domain)."""
    try:
        r_src = compute_all_metrics(
            model_circuit=masked_model, model_full=model,
            dataloader=src_dl, tokenizer=tokenizer,
            candidate_answers=candidate_answers,
            max_batches=max_eval,
        )
        r_tgt = compute_all_metrics(
            model_circuit=masked_model, model_full=model,
            dataloader=tgt_dl, tokenizer=tokenizer,
            candidate_answers=candidate_answers,
            max_batches=max_eval,
        )
        acc_src = r_src.get("circuit", r_src.get("full_model", {})).get("Acc", 0)
        acc_tgt = r_tgt.get("circuit", r_tgt.get("full_model", {})).get("Acc", 0)
        return acc_src - acc_tgt
    except Exception as e:
        logger.warning(f"Drop computation failed: {e}")
        return float("nan")


def run_ablation(backbone_key="llama2_7b", dataset_keys=None,
                 max_stage1_batches=None, max_stage2_batches=None,
                 max_eval_batches=None):
    _set_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset_keys = dataset_keys or list(DATASETS.keys())
    rows7, rows8, skipped = [], [], []

    try:
        model, tokenizer, rev = load_model_and_tokenizer(backbone_key)
    except Exception as e:
        logger.error(f"SKIPPED ablation — cannot load {backbone_key}: {e}")
        return [], []

    for dk in dataset_keys:
        logger.info(f"\n{'='*60}\nAblation: dataset={dk}\n{'='*60}")

        try:
            train_dl, val_dl, test_dl = load_spatial_dataset(dk)
        except Exception as e:
            msg = f"SKIPPED dataset {dk}: {e}"
            logger.error(msg)
            skipped.append(msg)
            continue

        cands = []
        for b in train_dl:
            cands.extend(b["label"])
        candidate_answers = _infer_candidates(cands)

        param_groups = get_parameter_groups(model)
        base_acc     = 0.85  # approximate; will be measured in "base" variant

        # Load the other dataset's test set for cross-domain drop
        other_dk = [k for k in DATASETS if k != dk][0]
        try:
            _, _, other_test = load_spatial_dataset(other_dk)
        except Exception:
            other_test = None

        for variant in ABLATION_VARIANTS:
            logger.info(f"  Running variant: {variant}")
            masked_model, metrics = _run_variant(
                variant, model, tokenizer, param_groups,
                train_dl, val_dl, test_dl,
                candidate_answers, base_acc,
                max_s1=max_stage1_batches,
                max_s2=max_stage2_batches,
                max_eval=max_eval_batches,
            )
            if not metrics:
                skipped.append(f"SKIPPED ablation variant={variant} dataset={dk}")
                continue

            if variant == "base":
                base_acc = metrics.get("Acc", base_acc) / 100.0

            rows7.append({
                "variant": variant, "dataset": dk,
                "Acc(%)":  f"{metrics.get('Acc',  'SKIPPED')}",
                "IR(%)":   f"{metrics.get('IR',   'SKIPPED')}",
                "CC(%)":   f"{metrics.get('CC',   'SKIPPED')}",
                "PE(%)":   f"{metrics.get('PE',   'SKIPPED')}",
                "AR(%)":   f"{metrics.get('AR',   'SKIPPED')}",
                "OS(%)":   f"{metrics.get('OS',   'SKIPPED')}",
            })

            drop = float("nan")
            if masked_model is not None and other_test is not None:
                drop = _compute_drop(model, masked_model, tokenizer,
                                     test_dl, other_test, candidate_answers,
                                     max_eval=max_eval_batches)

            rows8.append({
                "variant":   variant, "dataset": dk,
                "CC(%)":     f"{metrics.get('CC',        'SKIPPED')}",
                "Stability": f"{metrics.get('Stability', 'SKIPPED')}",
                "Var_int":   f"{metrics.get('Var_int',   'SKIPPED')}",
                "Drop(%)":   f"{drop:.1f}" if not (isinstance(drop, float) and
                                                   np.isnan(drop)) else "SKIPPED",
            })

            logger.info(f"    {variant}/{dk}: {metrics}")

    for fname, rows in [("table7.csv", rows7), ("table8.csv", rows8)]:
        path = os.path.join(RESULTS_DIR, fname)
        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            with open(path.replace(".csv", ".json"), "w") as f:
                json.dump(rows, f, indent=2)
            logger.info(f"Saved {path}")

    skip_path = os.path.join(RESULTS_DIR, "skipped_experiments.log")
    if skipped:
        with open(skip_path, "a") as f:
            f.write("\n=== Ablation ===\n")
            f.writelines(s + "\n" for s in skipped)

    return rows7, rows8
