"""Baseline method evaluations: PISTAQ, SREQA, NSM, PostGIS, GeoQA."""

import csv
import json
import logging
import os
import random

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (BACKBONE_MODELS, DATASETS, RESULTS_DIR, SEED,
                    MAX_SEQ_LEN)
from data.loader import load_spatial_dataset
from data.interventions import build_interventions
from metrics.evaluate import (accuracy, intervention_robustness,
                               causal_consistency, overall_score,
                               _infer_candidates)

logger = logging.getLogger(__name__)


PISTAQ_SYSTEM = (
    "You are a spatial reasoning system that extracts explicit spatial relations "
    "step-by-step (PISTAQ style). For each question, first list all spatial "
    "relations in the context, then compose them to answer.\n"
)

SREQA_SYSTEM = (
    "You are a spatial question answering system (SREQA style). Identify the "
    "key spatial entities and their relations, then apply compositional rules "
    "to infer the answer. Show your inference chain.\n"
)

NSM_SYSTEM = (
    "You are a neuro-symbolic spatial reasoner (NSM style). Parse the spatial "
    "context into a symbolic relation graph, then traverse the graph to answer "
    "the question. Output only the final answer.\n"
)

POSTGIS_SYSTEM = (
    "You are a spatial database engine (PostGIS style). Treat the context as "
    "a set of geometric predicates. Answer the question by applying topological "
    "and directional rules strictly.\n"
)

GEOQA_SYSTEM = (
    "You are a geospatial question answering system (GeoQA style). Use "
    "geographic knowledge and probabilistic spatial reasoning to select the best answer.\n"
)

BASELINE_PROMPTS = {
    "PISTAQ":  PISTAQ_SYSTEM,
    "SREQA":   SREQA_SYSTEM,
    "NSM":     NSM_SYSTEM,
    "PostGIS": POSTGIS_SYSTEM,
    "GeoQA":   GEOQA_SYSTEM,
}


def _run_baseline_method(model, tokenizer, test_dl, candidate_answers,
                         system_prompt, k_interv=1):
    model.eval()
    preds, preds_interv, labels_out = [], [], []

    with torch.no_grad():
        for batch in test_dl:
            for text, label in zip(batch["input_text"], batch["label"]):
                labels_out.append(label)

                best_pred, best_score = None, None
                for ans in candidate_answers:
                    prompt = f"{system_prompt}{text}\nAnswer: {ans}"
                    enc = tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=MAX_SEQ_LEN,
                        truncation=True,
                    ).to(model.device)
                    out = model(**enc, labels=enc["input_ids"])
                    score = -out.loss.item()
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pred  = ans
                preds.append(best_pred)

                perturbed = build_interventions(text, k=k_interv)[0]
                best_pred_i, best_score_i = None, None
                for ans in candidate_answers:
                    prompt_i = f"{system_prompt}{perturbed}\nAnswer: {ans}"
                    enc_i = tokenizer(
                        prompt_i,
                        return_tensors="pt",
                        max_length=MAX_SEQ_LEN,
                        truncation=True,
                    ).to(model.device)
                    out_i = model(**enc_i, labels=enc_i["input_ids"])
                    score_i = -out_i.loss.item()
                    if best_score_i is None or score_i > best_score_i:
                        best_score_i = score_i
                        best_pred_i  = ans
                preds_interv.append(best_pred_i)

    return preds, preds_interv, labels_out


def run_baselines(backbone_key="llama2_7b", dataset_keys=None):
    dataset_keys = dataset_keys or list(DATASETS.keys())

    try:
        from models.backbone import load_model_and_tokenizer
        model, tokenizer, _ = load_model_and_tokenizer(backbone_key)
    except Exception as e:
        logger.error(f"Cannot load backbone for baselines ({e}) — skipping baselines.")
        return []

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, skipped = [], []

    for dk in dataset_keys:
        try:
            _, _, test_dl = load_spatial_dataset(dk)
        except Exception as e:
            skipped.append(f"SKIPPED baseline dataset {dk}: {e}")
            continue

        cands = []
        for b in test_dl:
            cands.extend(b["label"])
        candidate_answers = _infer_candidates(cands)

        for method_name, system_prompt in BASELINE_PROMPTS.items():
            logger.info(f"  Running {method_name} on {dk} …")
            try:
                preds, preds_interv, labels = _run_baseline_method(
                    model, tokenizer, test_dl, candidate_answers, system_prompt
                )
                acc = accuracy(preds, labels)
                ir  = intervention_robustness(preds_interv, labels)
                cc  = causal_consistency(preds, preds_interv)
                pe  = 100.0   # full model used
                os_ = overall_score(acc, ir, pe)

                row = {
                    "method":   f"{method_name} (LLM approx.)",
                    "dataset":  dk,
                    "variant":  "Baseline",
                    "Acc":      f"{acc:.2f}",
                    "IR":       f"{ir:.2f}",
                    "CC":       f"{cc:.2f}",
                    "PE":       f"{pe:.1f}",
                    "AR":       "—",
                    "OS":       f"{os_:.2f}",
                }
                rows.append(row)
                logger.info(f"    {method_name}/{dk}: Acc={acc:.1f}, IR={ir:.1f}, "
                            f"CC={cc:.1f}, OS={os_:.1f}")

            except Exception as e:
                msg = f"SKIPPED baseline {method_name}/{dk}: {e}"
                logger.error(msg)
                skipped.append(msg)

    csv_path  = os.path.join(RESULTS_DIR, "baselines.csv")
    json_path = os.path.join(RESULTS_DIR, "baselines.json")
    skip_path = os.path.join(RESULTS_DIR, "skipped_experiments.log")

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Baselines saved to {csv_path}")

    if skipped:
        with open(skip_path, "a") as f:
            f.write("\n=== Baselines ===\n")
            f.writelines(s + "\n" for s in skipped)

    del model
    return rows
