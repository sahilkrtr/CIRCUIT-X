"""In-domain results on SPARTQA and StepGame."""

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
from models.backbone import (load_model_and_tokenizer, get_parameter_groups,
                              predict_answers)
from metrics.evaluate import compute_all_metrics, _infer_candidates
from stages.stage1 import estimate_importance
from stages.stage2 import discover_circuit
from baselines.run_baselines import run_baselines

logger = logging.getLogger(__name__)


def _set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def run_table4(backbone_keys=None, dataset_keys=None,
               max_stage1_batches=None, max_stage2_batches=None,
               max_eval_batches=None):
    _set_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    backbone_keys = backbone_keys or list(BACKBONE_MODELS.keys())
    dataset_keys  = dataset_keys  or list(DATASETS.keys())

    rows   = []
    skipped = []

    for bk in backbone_keys:
        try:
            model, tokenizer, rev = load_model_and_tokenizer(bk)
        except Exception as e:
            msg = f"SKIPPED model {bk}: {e}"
            logger.error(msg)
            skipped.append(msg)
            continue

        for dk in dataset_keys:
            logger.info(f"\n{'='*60}")
            logger.info(f"Table 4: backbone={bk}, dataset={dk}")
            logger.info(f"{'='*60}")

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

            logger.info("Evaluating base model …")
            try:
                base_results = compute_all_metrics(
                    model_circuit=None,
                    model_full=model,
                    dataloader=test_dl,
                    tokenizer=tokenizer,
                    candidate_answers=candidate_answers,
                    max_batches=max_eval_batches,
                )
                base_metrics = base_results["full_model"]
                logger.info(f"Base metrics: {base_metrics}")
            except Exception as e:
                msg = f"SKIPPED base eval {bk}/{dk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                continue

            rows.append({
                "method": bk, "dataset": dk, "variant": "Base",
                **{k: f"{v:.2f}" for k, v in base_metrics.items()},
            })

            stage1_cache = os.path.join(CHECKPOINT_DIR, bk, dk, "stage1_cache.json")
            param_groups = get_parameter_groups(model)

            if os.path.exists(stage1_cache):
                logger.info(f"Loading Stage I results from cache: {stage1_cache}")
                try:
                    with open(stage1_cache) as f:
                        cache = json.load(f)
                    selected_names = set(cache["selected_names"])
                    theta_s = [g for g in param_groups if g["name"] in selected_names]
                    stage1_scores = cache.get("scores", {})
                    logger.info(f"Stage I (cached): {len(theta_s)}/{len(param_groups)} groups selected")
                except Exception as e:
                    logger.warning(f"Stage I cache load failed ({e}), re-running Stage I")
                    theta_s = None
            else:
                theta_s = None

            if theta_s is None:
                logger.info("Running Stage I …")
                try:
                    theta_s, stage1_scores = estimate_importance(
                        model, tokenizer, param_groups, train_dl,
                        tau1=TAU1, tau2=TAU2,
                        max_batches=max_stage1_batches or BATCH_SIZE_STAGE1,
                    )
                    os.makedirs(os.path.dirname(stage1_cache), exist_ok=True)
                    with open(stage1_cache, "w") as f:
                        json.dump({
                            "selected_names": [g["name"] for g in theta_s],
                            "scores": stage1_scores,
                            "tau1": TAU1, "tau2": TAU2,
                            "max_batches": max_stage1_batches,
                        }, f, indent=2)
                    logger.info(f"Stage I results cached to {stage1_cache}")
                except Exception as e:
                    msg = f"SKIPPED Stage I {bk}/{dk}: {e}"
                    logger.error(msg)
                    skipped.append(msg)
                    continue

            if not theta_s:
                msg = f"SKIPPED Stage II {bk}/{dk}: θ_s empty after Stage I"
                logger.warning(msg)
                skipped.append(msg)
                continue

            torch.cuda.empty_cache()
            logger.info("Running Stage II …")
            ckpt_path = os.path.join(CHECKPOINT_DIR, bk, dk)
            try:
                circuit_mask, masked_model = discover_circuit(
                    model, tokenizer, theta_s,
                    train_dl, val_dl, candidate_answers,
                    base_accuracy=base_metrics["Acc"] / 100.0,
                    lambda1=LAMBDA1, lambda2=LAMBDA2,
                    lr=LR_STAGE2, epochs=EPOCHS_STAGE2, epsilon=EPSILON,
                    max_batches_per_epoch=max_stage2_batches,
                    checkpoint_path=ckpt_path,
                )
            except Exception as e:
                msg = f"SKIPPED Stage II {bk}/{dk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                continue

            logger.info("Evaluating CIRCUIT-X model …")
            try:
                cx_results = compute_all_metrics(
                    model_circuit=masked_model,
                    model_full=model,
                    dataloader=test_dl,
                    tokenizer=tokenizer,
                    candidate_answers=candidate_answers,
                    max_batches=max_eval_batches,
                )
                cx_metrics = cx_results["circuit"]
                logger.info(f"CIRCUIT-X metrics: {cx_metrics}")
            except Exception as e:
                msg = f"SKIPPED CX eval {bk}/{dk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                continue

            rows.append({
                "method": f"{bk}+CIRCUIT-X", "dataset": dk, "variant": "CIRCUIT-X",
                **{k: f"{v:.2f}" for k, v in cx_metrics.items()},
            })

        del model

    logger.info("Running baselines …")
    baseline_rows = run_baselines(dataset_keys=dataset_keys)
    rows.extend(baseline_rows)

    csv_path  = os.path.join(RESULTS_DIR, "table4.csv")
    json_path = os.path.join(RESULTS_DIR, "table4.json")
    skip_path = os.path.join(RESULTS_DIR, "skipped_experiments.log")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Table 4 saved to {csv_path}")

    if skipped:
        with open(skip_path, "a") as f:
            f.write("\n=== Table 4 ===\n")
            f.writelines(s + "\n" for s in skipped)

    return rows
