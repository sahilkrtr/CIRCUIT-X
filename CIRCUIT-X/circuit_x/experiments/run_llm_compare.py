"""Comparison with instruction-tuned and reasoning LLMs."""

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
                    CAN_RUN_70B, TOTAL_VRAM_GB)
from data.loader import load_spatial_dataset
from models.backbone import load_model_and_tokenizer, get_parameter_groups
from metrics.evaluate import compute_all_metrics, _infer_candidates
from stages.stage1 import estimate_importance
from stages.stage2 import discover_circuit

logger = logging.getLogger(__name__)

LOCAL_MODELS = {
    "llama2_7b":        "meta-llama/Llama-2-7b-hf",
    "mistral_7b":       "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma_7b":         "google/gemma-7b-it",
}

API_MODELS = {
    "gpt4":             ("openai",     "gpt-4",            "OPENAI_API_KEY"),
    "claude3_opus":     ("anthropic",  "claude-3-opus-20240229", "ANTHROPIC_API_KEY"),
    "gemini1_5_pro":    ("google",     "gemini-1.5-pro",   "GOOGLE_API_KEY"),
    "palm2":            ("google",     "text-bison",        "GOOGLE_API_KEY"),
}

LARGE_LOCAL_MODELS = {
    "qwen2_72b":        "Qwen/Qwen2-72B-Instruct",
    "deepseek_67b":     "deepseek-ai/deepseek-llm-67b-base",
    "llama3_70b":       "meta-llama/Meta-Llama-3-70B-Instruct",
}


def _set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def _eval_api_model(model_id, provider, api_key_env, dataset_key, tokenizer=None):
    """Score a cloud model via its API (returns metrics dict or None)."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        logger.warning(f"API key {api_key_env} not set — skipping {model_id}.")
        return None

    if provider == "openai":
        return _eval_openai(model_id, api_key, dataset_key)
    elif provider == "anthropic":
        return _eval_anthropic(model_id, api_key, dataset_key)
    elif provider == "google":
        return _eval_google(model_id, api_key, dataset_key)
    return None


def _eval_openai(model_id, api_key, dataset_key):
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai package not installed — skipping GPT-4.")
        return None

    from data.loader import load_spatial_dataset
    from data.interventions import build_interventions
    _, _, test_dl = load_spatial_dataset(dataset_key)
    cands = []
    for b in test_dl:
        cands.extend(b["label"])
    candidate_answers = _infer_candidates(cands)

    preds, preds_interv, labels = [], [], []
    for batch in test_dl:
        for text, label in zip(batch["input_text"], batch["label"]):
            labels.append(label)
            prompt = (f"Choose the best answer.\n\n{text}\n\n"
                      f"Options: {', '.join(candidate_answers)}\nAnswer:")
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                )
                pred = resp.choices[0].message.content.strip()
            except Exception:
                pred = candidate_answers[0]
            preds.append(pred)

            perturbed = build_interventions(text, k=1)[0]
            prompt_i = (f"Choose the best answer.\n\n{perturbed}\n\n"
                        f"Options: {', '.join(candidate_answers)}\nAnswer:")
            try:
                resp_i = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_i}],
                    max_tokens=10,
                )
                pred_i = resp_i.choices[0].message.content.strip()
            except Exception:
                pred_i = candidate_answers[0]
            preds_interv.append(pred_i)

    from metrics.evaluate import (accuracy, intervention_robustness,
                                  causal_consistency, overall_score)
    acc = accuracy(preds, labels)
    ir  = intervention_robustness(preds_interv, labels)
    cc  = causal_consistency(preds, preds_interv)
    return {"Acc": acc, "IR": ir, "CC": cc, "PE": "—", "AR": "—",
            "OS": overall_score(acc, ir, 100.0)}


def _eval_anthropic(model_id, api_key, dataset_key):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.warning("anthropic package not installed — skipping Claude.")
        return None

    from data.loader import load_spatial_dataset
    from data.interventions import build_interventions
    _, _, test_dl = load_spatial_dataset(dataset_key)
    cands = []
    for b in test_dl:
        cands.extend(b["label"])
    candidate_answers = _infer_candidates(cands)

    preds, preds_interv, labels = [], [], []
    for batch in test_dl:
        for text, label in zip(batch["input_text"], batch["label"]):
            labels.append(label)
            prompt = (f"Choose the best answer.\n\n{text}\n\n"
                      f"Options: {', '.join(candidate_answers)}\nAnswer:")
            try:
                msg = client.messages.create(
                    model=model_id,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                pred = msg.content[0].text.strip()
            except Exception:
                pred = candidate_answers[0]
            preds.append(pred)

            perturbed = build_interventions(text, k=1)[0]
            prompt_i  = (f"Choose the best answer.\n\n{perturbed}\n\n"
                         f"Options: {', '.join(candidate_answers)}\nAnswer:")
            try:
                msg_i = client.messages.create(
                    model=model_id, max_tokens=10,
                    messages=[{"role": "user", "content": prompt_i}],
                )
                pred_i = msg_i.content[0].text.strip()
            except Exception:
                pred_i = candidate_answers[0]
            preds_interv.append(pred_i)

    from metrics.evaluate import (accuracy, intervention_robustness,
                                  causal_consistency, overall_score)
    acc = accuracy(preds, labels)
    ir  = intervention_robustness(preds_interv, labels)
    cc  = causal_consistency(preds, preds_interv)
    return {"Acc": acc, "IR": ir, "CC": cc, "PE": "—", "AR": "—",
            "OS": overall_score(acc, ir, 100.0)}


def _eval_google(model_id, api_key, dataset_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model_id)
    except ImportError:
        logger.warning("google-generativeai package not installed — skipping Gemini/PaLM.")
        return None

    from data.loader import load_spatial_dataset
    from data.interventions import build_interventions
    _, _, test_dl = load_spatial_dataset(dataset_key)
    cands = []
    for b in test_dl:
        cands.extend(b["label"])
    candidate_answers = _infer_candidates(cands)

    preds, preds_interv, labels = [], [], []
    for batch in test_dl:
        for text, label in zip(batch["input_text"], batch["label"]):
            labels.append(label)
            prompt = (f"Choose the best answer.\n\n{text}\n\n"
                      f"Options: {', '.join(candidate_answers)}\nAnswer:")
            try:
                resp = gmodel.generate_content(prompt)
                pred = resp.text.strip()
            except Exception:
                pred = candidate_answers[0]
            preds.append(pred)

            perturbed = build_interventions(text, k=1)[0]
            try:
                resp_i = gmodel.generate_content(
                    f"Choose the best answer.\n\n{perturbed}\n\n"
                    f"Options: {', '.join(candidate_answers)}\nAnswer:")
                pred_i = resp_i.text.strip()
            except Exception:
                pred_i = candidate_answers[0]
            preds_interv.append(pred_i)

    from metrics.evaluate import (accuracy, intervention_robustness,
                                  causal_consistency, overall_score)
    acc = accuracy(preds, labels)
    ir  = intervention_robustness(preds_interv, labels)
    cc  = causal_consistency(preds, preds_interv)
    return {"Acc": acc, "IR": ir, "CC": cc, "PE": "—", "AR": "—",
            "OS": overall_score(acc, ir, 100.0)}


def run_llm_compare(dataset_keys=None, max_stage1_batches=None,
                    max_stage2_batches=None):
    _set_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset_keys = dataset_keys or list(DATASETS.keys())

    rows, skipped = [], []

    for bk, hf_name in LOCAL_MODELS.items():
        for dk in dataset_keys:
            try:
                train_dl, val_dl, test_dl = load_spatial_dataset(dk)
            except Exception as e:
                skipped.append(f"SKIPPED {bk}/{dk}: {e}")
                continue

            cands = []
            for b in train_dl:
                cands.extend(b["label"])
            candidate_answers = _infer_candidates(cands)

            try:
                model, tokenizer, _ = load_model_and_tokenizer(bk)
            except Exception as e:
                skipped.append(f"SKIPPED model {bk}: {e}")
                continue

            try:
                base_res = compute_all_metrics(
                    None, model, test_dl, tokenizer, candidate_answers)
                bm = base_res["full_model"]
                rows.append({"method": bk, "dataset": dk, "variant": "Base",
                              **{k: f"{v:.2f}" for k, v in bm.items()}})
            except Exception as e:
                skipped.append(f"SKIPPED base {bk}/{dk}: {e}")
                del model
                continue

            try:
                pg = get_parameter_groups(model)
                theta_s, _ = estimate_importance(
                    model, tokenizer, pg, train_dl,
                    tau1=TAU1, tau2=TAU2,
                    max_batches=max_stage1_batches or BATCH_SIZE_STAGE1)
                if theta_s:
                    _, mm = discover_circuit(
                        model, tokenizer, theta_s, train_dl, val_dl,
                        candidate_answers,
                        base_accuracy=bm["Acc"] / 100.0,
                        lambda1=LAMBDA1, lambda2=LAMBDA2,
                        lr=LR_STAGE2, epochs=EPOCHS_STAGE2, epsilon=EPSILON,
                        max_batches_per_epoch=max_stage2_batches)
                    cx_res = compute_all_metrics(mm, model, test_dl, tokenizer,
                                                 candidate_answers)
                    cxm = cx_res["circuit"]
                    rows.append({"method": f"{bk}+CIRCUIT-X", "dataset": dk,
                                 "variant": "CIRCUIT-X",
                                 **{k: f"{v:.2f}" for k, v in cxm.items()}})
            except Exception as e:
                skipped.append(f"SKIPPED CX {bk}/{dk}: {e}")
            del model

    for api_key_name, (provider, model_id, api_key_env) in API_MODELS.items():
        for dk in dataset_keys:
            metrics = _eval_api_model(model_id, provider, api_key_env, dk)
            if metrics is None:
                skipped.append(f"SKIPPED API model {api_key_name}/{dk}: key not set")
                continue
            rows.append({"method": api_key_name, "dataset": dk, "variant": "Base",
                         **{k: f"{v}" for k, v in metrics.items()}})

    logger.info(f"Total VRAM: {TOTAL_VRAM_GB:.1f} GB | CAN_RUN_70B={CAN_RUN_70B}")

    if CAN_RUN_70B:
        for bk, hf_name in LARGE_LOCAL_MODELS.items():
            for dk in dataset_keys:
                try:
                    train_dl, val_dl, test_dl = load_spatial_dataset(dk)
                    cands = []
                    for b in train_dl:
                        cands.extend(b["label"])
                    candidate_answers = _infer_candidates(cands)
                    model, tokenizer, _ = load_model_and_tokenizer(
                        bk, quantize_4bit=True)
                    base_res = compute_all_metrics(None, model, test_dl, tokenizer,
                                                   candidate_answers)
                    bm = base_res["full_model"]
                    rows.append({"method": bk, "dataset": dk, "variant": "Base",
                                 **{k: f"{v:.2f}" for k, v in bm.items()}})
                    del model
                except Exception as e:
                    skipped.append(f"SKIPPED large model {bk}/{dk}: {e}")
    else:
        logger.info(f"Large models skipped: total VRAM={TOTAL_VRAM_GB:.1f} GB < 40 GB required.")
        for bk in LARGE_LOCAL_MODELS:
            for dk in dataset_keys:
                skipped.append(f"SKIPPED large model {bk}/{dk}: insufficient VRAM "
                               f"({TOTAL_VRAM_GB:.1f} GB total)")

    csv_path  = os.path.join(RESULTS_DIR, "table10.csv")
    json_path = os.path.join(RESULTS_DIR, "table10.json")
    skip_path = os.path.join(RESULTS_DIR, "skipped_experiments.log")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Table 10 saved to {csv_path}")

    if skipped:
        with open(skip_path, "a") as f:
            f.write("\n=== Table 10 ===\n")
            f.writelines(s + "\n" for s in skipped)

    return rows
