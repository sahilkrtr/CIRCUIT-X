"""Real-world geoinformatics evaluation on dataset_geography."""

import csv
import json
import logging
import os
import random

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (BACKBONE_MODELS, GEO_DATASET_HF, RESULTS_DIR, CHECKPOINT_DIR,
                    SEED, TAU1, TAU2, LAMBDA1, LAMBDA2, LR_STAGE2,
                    EPOCHS_STAGE2, EPSILON, BATCH_SIZE_STAGE1)
from data.loader import load_geo_dataset
from models.backbone import load_model_and_tokenizer, get_parameter_groups
from metrics.evaluate import compute_all_metrics, _infer_candidates
from stages.stage1 import estimate_importance
from stages.stage2 import discover_circuit

logger = logging.getLogger(__name__)


def _set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def run_geoeval(backbone_keys=None, max_stage1_batches=None,
                max_stage2_batches=None, max_eval_batches=None):
    _set_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    backbone_keys = backbone_keys or list(BACKBONE_MODELS.keys())
    rows, skipped = [], []

    try:
        train_dl, val_dl, test_dl = load_geo_dataset(GEO_DATASET_HF)
    except Exception as e:
        logger.error(f"Cannot load geo dataset {GEO_DATASET_HF}: {e}")
        return []

    cands = []
    for b in train_dl:
        cands.extend(b["label"])
    candidate_answers = _infer_candidates(cands)

    for bk in backbone_keys:
        logger.info(f"\n{'='*60}\nGeoEval: backbone={bk}\n{'='*60}")

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
                dataloader=test_dl, tokenizer=tokenizer,
                candidate_answers=candidate_answers,
                max_batches=max_eval_batches,
            )
            base_metrics = base_results["full_model"]
            logger.info(f"Base {bk}: {base_metrics}")
        except Exception as e:
            msg = f"SKIPPED base geo eval {bk}: {e}"
            logger.error(msg)
            skipped.append(msg)
            del model
            continue

        rows.append({
            "method": bk, "variant": "Base",
            **{k: f"{v:.2f}" for k, v in base_metrics.items()},
        })

        try:
            param_groups = get_parameter_groups(model)
            theta_s, _ = estimate_importance(
                model, tokenizer, param_groups, train_dl,
                tau1=TAU1, tau2=TAU2,
                max_batches=max_stage1_batches or BATCH_SIZE_STAGE1,
            )
        except Exception as e:
            msg = f"SKIPPED Stage I geo {bk}: {e}"
            logger.error(msg)
            skipped.append(msg)
            del model
            continue

        if not theta_s:
            msg = f"SKIPPED Stage II geo {bk}: empty θ_s"
            logger.warning(msg)
            skipped.append(msg)
            del model
            continue

        torch.cuda.empty_cache()
        ckpt_path = os.path.join(CHECKPOINT_DIR, bk, "geo")
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
            msg = f"SKIPPED Stage II geo {bk}: {e}"
            logger.error(msg)
            skipped.append(msg)
            del model
            continue

        try:
            cx_results = compute_all_metrics(
                model_circuit=masked_model, model_full=model,
                dataloader=test_dl, tokenizer=tokenizer,
                candidate_answers=candidate_answers,
                max_batches=max_eval_batches,
            )
            cx_metrics = cx_results["circuit"]
            logger.info(f"CIRCUIT-X {bk} geo: {cx_metrics}")
        except Exception as e:
            msg = f"SKIPPED CX geo eval {bk}: {e}"
            logger.error(msg)
            skipped.append(msg)
            del model
            continue

        rows.append({
            "method": f"{bk}+CIRCUIT-X", "variant": "CIRCUIT-X",
            **{k: f"{v:.2f}" for k, v in cx_metrics.items()},
        })
        del model

    csv_path  = os.path.join(RESULTS_DIR, "table9.csv")
    json_path = os.path.join(RESULTS_DIR, "table9.json")
    skip_path = os.path.join(RESULTS_DIR, "skipped_experiments.log")

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Table 9 saved to {csv_path}")

    if skipped:
        with open(skip_path, "a") as f:
            f.write("\n=== Table 9 ===\n")
            f.writelines(s + "\n" for s in skipped)

    return rows
