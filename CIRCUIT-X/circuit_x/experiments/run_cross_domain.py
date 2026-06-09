"""Cross-domain transfer evaluation."""

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
                    EPOCHS_STAGE2, EPSILON, BATCH_SIZE_STAGE1)
from data.loader import load_spatial_dataset
from models.backbone import load_model_and_tokenizer, get_parameter_groups
from metrics.evaluate import compute_all_metrics, _infer_candidates
from stages.stage1 import estimate_importance
from stages.stage2 import discover_circuit

logger = logging.getLogger(__name__)


def _set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


TRANSFER_PAIRS = [
    ("spartqa", "stepgame"),   # train on SPARTQA, test on StepGame
    ("stepgame", "spartqa"),   # train on StepGame, test on SPARTQA
]


def run_cross_domain(backbone_keys=None, max_stage1_batches=None,
                     max_stage2_batches=None, max_eval_batches=None):
    _set_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    backbone_keys = backbone_keys or list(BACKBONE_MODELS.keys())
    rows, skipped = [], []

    for src_dk, tgt_dk in TRANSFER_PAIRS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-domain: train={src_dk} → test={tgt_dk}")
        logger.info(f"{'='*60}")

        try:
            src_train, src_val, _   = load_spatial_dataset(src_dk)
            _,         _,       tgt_test = load_spatial_dataset(tgt_dk)
        except Exception as e:
            msg = f"SKIPPED cross-domain {src_dk}→{tgt_dk}: {e}"
            logger.error(msg)
            skipped.append(msg)
            continue

        tgt_labels = []
        for b in tgt_test:
            tgt_labels.extend(b["label"])
        candidate_answers = _infer_candidates(tgt_labels)

        for bk in backbone_keys:
            try:
                model, tokenizer, rev = load_model_and_tokenizer(bk)
            except Exception as e:
                msg = f"SKIPPED model {bk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                continue

            try:
                base_results = compute_all_metrics(
                    model_circuit=None, model_full=model,
                    dataloader=tgt_test, tokenizer=tokenizer,
                    candidate_answers=candidate_answers,
                    max_batches=max_eval_batches,
                )
                base_metrics = base_results["full_model"]
                logger.info(f"Base ({bk}) train={src_dk}→test={tgt_dk}: {base_metrics}")
            except Exception as e:
                msg = f"SKIPPED base cross-domain eval {bk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                del model
                continue

            rows.append({
                "train": src_dk, "test": tgt_dk,
                "method": bk, "variant": "Base",
                **{k: f"{v:.2f}" for k, v in base_metrics.items()},
            })

            try:
                param_groups = get_parameter_groups(model)
                theta_s, _ = estimate_importance(
                    model, tokenizer, param_groups, src_train,
                    tau1=TAU1, tau2=TAU2,
                    max_batches=max_stage1_batches or BATCH_SIZE_STAGE1,
                )
            except Exception as e:
                msg = f"SKIPPED Stage I cross-domain {bk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                del model
                continue

            if not theta_s:
                msg = f"SKIPPED Stage II cross-domain {bk}: empty θ_s"
                logger.warning(msg)
                skipped.append(msg)
                del model
                continue

            torch.cuda.empty_cache()
            ckpt_path = os.path.join(CHECKPOINT_DIR, bk, f"{src_dk}_to_{tgt_dk}")
            try:
                circuit_mask, masked_model = discover_circuit(
                    model, tokenizer, theta_s,
                    src_train, src_val, candidate_answers,
                    base_accuracy=base_metrics["Acc"] / 100.0,
                    lambda1=LAMBDA1, lambda2=LAMBDA2,
                    lr=LR_STAGE2, epochs=EPOCHS_STAGE2, epsilon=EPSILON,
                    max_batches_per_epoch=max_stage2_batches,
                    checkpoint_path=ckpt_path,
                )
            except Exception as e:
                msg = f"SKIPPED Stage II cross-domain {bk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                del model
                continue

            try:
                cx_results = compute_all_metrics(
                    model_circuit=masked_model, model_full=model,
                    dataloader=tgt_test, tokenizer=tokenizer,
                    candidate_answers=candidate_answers,
                    max_batches=max_eval_batches,
                )
                cx_metrics = cx_results["circuit"]
                logger.info(f"CIRCUIT-X ({bk}) train={src_dk}→test={tgt_dk}: {cx_metrics}")
            except Exception as e:
                msg = f"SKIPPED CX cross-domain eval {bk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                del model
                continue

            rows.append({
                "train": src_dk, "test": tgt_dk,
                "method": f"{bk}+CIRCUIT-X", "variant": "CIRCUIT-X",
                **{k: f"{v:.2f}" for k, v in cx_metrics.items()},
            })

            del model

    csv_path  = os.path.join(RESULTS_DIR, "table5.csv")
    json_path = os.path.join(RESULTS_DIR, "table5.json")
    skip_path = os.path.join(RESULTS_DIR, "skipped_experiments.log")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Table 5 saved to {csv_path}")

    if skipped:
        with open(skip_path, "a") as f:
            f.write("\n=== Table 5 ===\n")
            f.writelines(s + "\n" for s in skipped)

    return rows
