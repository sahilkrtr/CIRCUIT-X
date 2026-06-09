"""Evaluation metrics and batched log-probability scoring."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MAX_SEQ_LEN, MC_SAMPLES_K
from data.interventions import build_interventions

logger = logging.getLogger(__name__)


def accuracy(predictions: List[str], labels: List[str]) -> float:
    """Exact match accuracy in %."""
    if not labels:
        return 0.0
    correct = sum(
        str(p).strip().lower() == str(l).strip().lower()
        for p, l in zip(predictions, labels)
    )
    return 100.0 * correct / len(labels)


def intervention_robustness(predictions_interv: List[str], labels: List[str]) -> float:
    """Accuracy on intervened inputs in %."""
    return accuracy(predictions_interv, labels)


def causal_consistency(predictions_original: List[str],
                       predictions_intervened: List[str]) -> float:
    """Fraction of samples where prediction is unchanged after intervention, in %."""
    if not predictions_original:
        return 0.0
    inconsistent = sum(
        str(p).strip().lower() != str(pi).strip().lower()
        for p, pi in zip(predictions_original, predictions_intervened)
    )
    return 100.0 * (1.0 - inconsistent / len(predictions_original))


def parameter_efficiency(theta_c_params: int, theta_full_params: int) -> float:
    """Circuit parameters as a fraction of total parameters, in %."""
    if theta_full_params == 0:
        return 0.0
    return 100.0 * theta_c_params / theta_full_params


def accuracy_retention(acc_circuit: float, acc_full: float) -> float:
    """Circuit accuracy divided by full-model accuracy (ratio)."""
    if acc_full == 0.0:
        return 0.0
    return acc_circuit / acc_full


def overall_score(acc: float, ir: float, pe: float) -> float:
    """Combined score: Acc × IR × (1 − PE). All inputs in % (0–100)."""
    return 100.0 * (acc / 100.0) * (ir / 100.0) * (1.0 - pe / 100.0)


def prediction_stability(per_sample_preds: List[List[str]]) -> float:
    """Fraction of samples where prediction is identical across all K perturbations."""
    if not per_sample_preds:
        return 0.0
    stable = sum(
        len(set(str(p).strip().lower() for p in preds)) == 1
        for preds in per_sample_preds
    )
    return 100.0 * stable / len(per_sample_preds)


def variance_under_intervention(per_sample_probs: List[List[float]]) -> float:
    """Mean variance of prediction probability across K intervention samples."""
    if not per_sample_probs:
        return 0.0
    return float(np.mean([np.var(probs) for probs in per_sample_probs]))


def _score_sequences(model, tokenizer, prompts: List[str]) -> List[float]:
    """Return per-sequence NLL score (higher = more probable) for a list of prompts.

    Runs ONE forward pass for the whole list, extracting per-sample loss from
    the shifted logits — no padding token contributes to the loss.
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=True,
        return_attention_mask=True,
    )
    # Move to the model's first device (device_map="auto" may spread layers)
    device = next(model.parameters()).device
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out    = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # (B, T, V)

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()     # (B, T-1)
    shift_mask   = attention_mask[:, 1:].contiguous().float()

    loss_fct      = nn.CrossEntropyLoss(reduction="none")
    per_tok_loss  = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).reshape(shift_labels.size())                   # (B, T-1)

    # Mask padding; average over non-padding tokens
    masked_loss   = per_tok_loss * shift_mask
    seq_lengths   = shift_mask.sum(dim=1).clamp(min=1)
    per_seq_nll   = masked_loss.sum(dim=1) / seq_lengths  # (B,)

    return (-per_seq_nll).tolist()   # higher score = lower NLL = more probable


def _predict_batch(model, tokenizer,
                   texts: List[str],
                   candidate_answers: List[str]) -> List[Tuple[str, float]]:
    """Score every (text, answer) pair in batch; return (best_answer, prob) per text.

    For N_cands answers and B texts:
      - runs N_cands forward passes, each processing B samples
      - much faster than B × N_cands sequential single-sample passes
    """
    # all_scores[ci][ti] = score for text ti under answer candidate ci
    all_scores: List[List[float]] = []
    for ans in candidate_answers:
        prompts = [f"{t}\nAnswer: {ans}" for t in texts]
        all_scores.append(_score_sequences(model, tokenizer, prompts))

    results: List[Tuple[str, float]] = []
    n_cands = len(candidate_answers)
    for ti in range(len(texts)):
        cand_scores = torch.tensor([all_scores[ci][ti] for ci in range(n_cands)])
        probs       = torch.softmax(cand_scores, dim=0)
        best_idx    = int(probs.argmax())
        results.append((candidate_answers[best_idx], probs[best_idx].item()))

    return results


def _infer_candidates(labels: List[str]) -> List[str]:
    """Derive candidate answer list from observed labels."""
    unique = list(dict.fromkeys(str(l).strip() for l in labels if str(l).strip()))
    return unique if unique else ["yes", "no", "A", "B", "C", "D"]


def compute_all_metrics(
    model_circuit,
    model_full,
    dataloader: DataLoader,
    tokenizer,
    candidate_answers: Optional[List[str]] = None,
    k: int = MC_SAMPLES_K,
    compute_stability: bool = False,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Run full evaluation and return dict with all 6 metrics.

    Uses batched log-probability scoring: N_answers forward passes per batch
    instead of B × N_answers sequential passes.

    Parameters
    ----------
    model_circuit   : CIRCUIT-X masked model (or None for full-model only)
    model_full      : original backbone
    dataloader      : test DataLoader
    tokenizer
    candidate_answers : auto-derived from labels if None
    k               : MC intervention samples for IR / CC
    compute_stability : also compute Stability and Var_int
    max_batches     : truncate evaluation (for smoke tests)
    """
    model_full.eval()
    if model_circuit is not None:
        model_circuit.eval()

    all_labels:            List[str]         = []
    preds_full:            List[str]         = []
    preds_circuit:         List[str]         = []
    preds_interv_full:     List[str]         = []
    preds_interv_circuit:  List[str]         = []
    stability_preds:       List[List[str]]   = []
    stability_probs:       List[List[float]] = []

    if candidate_answers is None:
        raw: List[str] = []
        for batch in dataloader:
            raw.extend(batch["label"])
        candidate_answers = _infer_candidates(raw)

    logger.info(f"Evaluating | candidates={candidate_answers} | "
                f"batched scoring (N_passes/batch={len(candidate_answers)})")

    n_done = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        texts  = batch["input_text"]
        labels = batch["label"]
        all_labels.extend(labels)

        full_results = _predict_batch(model_full, tokenizer, texts, candidate_answers)
        preds_full.extend(pred for pred, _ in full_results)

        first_perturbations = [build_interventions(t, k=1)[0] for t in texts]
        full_interv_results = _predict_batch(model_full, tokenizer,
                                             first_perturbations, candidate_answers)
        preds_interv_full.extend(pred for pred, _ in full_interv_results)

        if model_circuit is not None:
            circ_results = _predict_batch(model_circuit, tokenizer,
                                          texts, candidate_answers)
            preds_circuit.extend(pred for pred, _ in circ_results)

            # Intervention predictions for IR / CC
            circ_interv_results = _predict_batch(model_circuit, tokenizer,
                                                 first_perturbations, candidate_answers)
            preds_interv_circuit.extend(pred for pred, _ in circ_interv_results)

            if compute_stability:
                for text in texts:
                    perturbed = build_interventions(text, k=k)
                    k_results = _predict_batch(model_circuit, tokenizer,
                                               perturbed, candidate_answers)
                    stability_preds.append([p for p, _ in k_results])
                    stability_probs.append([prob for _, prob in k_results])

        n_done += 1
        if max_batches and n_done >= max_batches:
            break

    acc_full = accuracy(preds_full, all_labels)
    ir_full  = intervention_robustness(preds_interv_full, all_labels)
    cc_full  = causal_consistency(preds_full, preds_interv_full)

    results = {
        "full_model": {
            "Acc": acc_full,
            "IR":  ir_full,
            "CC":  cc_full,
            "PE":  100.0,
            "AR":  1.0,
            "OS":  overall_score(acc_full, ir_full, 100.0),
        }
    }

    if model_circuit is not None and preds_circuit:
        total_params  = sum(p.numel() for p in model_full.parameters())
        if hasattr(model_circuit, "circuit_mask"):
            cm = model_circuit.circuit_mask
            active_params = sum(
                sum(p.numel() for _, p in grp["params"])
                for gate, grp in zip(cm.hard_gates(), cm.groups)
                if gate == 1.0
            )
        else:
            active_params = total_params

        acc_circ = accuracy(preds_circuit, all_labels)
        ir_circ  = intervention_robustness(preds_interv_circuit, all_labels)
        cc_circ  = causal_consistency(preds_circuit, preds_interv_circuit)
        pe_circ  = parameter_efficiency(active_params, total_params)
        ar_circ  = accuracy_retention(acc_circ, acc_full)
        os_circ  = overall_score(acc_circ, ir_circ, pe_circ)

        results["circuit"] = {
            "Acc": acc_circ,
            "IR":  ir_circ,
            "CC":  cc_circ,
            "PE":  pe_circ,
            "AR":  ar_circ * 100.0,
            "OS":  os_circ,
        }
        if compute_stability:
            results["circuit"]["Stability"] = prediction_stability(stability_preds)
            results["circuit"]["Var_int"]   = variance_under_intervention(stability_probs)

    return results
