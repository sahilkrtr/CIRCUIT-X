"""Causal importance estimation for parameter group selection."""

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TAU1, TAU2, BATCH_SIZE_STAGE1, MC_SAMPLES_K, MAX_SEQ_LEN
from data.interventions import build_interventions
from models.backbone import ablated_params

logger = logging.getLogger(__name__)


def _get_loss(model, tokenizer, texts: List[str]) -> torch.Tensor:
    """Return per-sample NLL loss for a batch of texts."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=True,
        return_attention_mask=True,
    )
    device     = next(model.parameters()).device
    input_ids  = enc["input_ids"].to(device)
    attn_mask  = enc["attention_mask"].to(device)

    import torch.nn as nn_
    with torch.no_grad():
        out    = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        logits = out.logits   # (B, T, V)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask   = attn_mask[:, 1:].contiguous().float()
    del out, logits, input_ids, attn_mask  # free large GPU tensors immediately

    loss_fct     = nn_.CrossEntropyLoss(reduction="none")
    per_tok_loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).reshape(shift_labels.size())
    del shift_logits, shift_labels

    masked_loss  = per_tok_loss * shift_mask
    seq_len      = shift_mask.sum(dim=1).clamp(min=1)
    del per_tok_loss, shift_mask
    per_seq_nll  = masked_loss.sum(dim=1) / seq_len
    del masked_loss, seq_len
    result = per_seq_nll.cpu()
    del per_seq_nll
    return result


def estimate_importance(
    model,
    tokenizer,
    param_groups: List[Dict],
    dataloader: DataLoader,
    tau1: float = TAU1,
    tau2: float = TAU2,
    k: int = MC_SAMPLES_K,
    max_batches: Optional[int] = None,
) -> List[Dict]:
    """Score each parameter group by causal importance and return those above threshold."""
    model.eval()
    n_groups  = len(param_groups)
    delta_sum        = torch.zeros(n_groups)
    delta_causal_sum = torch.zeros(n_groups)
    n_batches = 0

    logger.info(f"Stage I: scoring {n_groups} parameter groups …")

    for batch in tqdm(dataloader, desc="Stage I batches"):
        texts = batch["input_text"]

        p_original = _get_loss(model, tokenizer, texts)  # shape (B,)

        all_perturbed = [build_interventions(t, k=k) for t in texts]

        # Transpose the intervention list: k passes of B samples each (avoids OOM from B*k)
        # all_perturbed is List[B][k] → transposed_perturbed is List[k][B]
        B = len(texts)
        k_actual = len(all_perturbed[0]) if all_perturbed else 0
        transposed = [[all_perturbed[b][ki] for b in range(B)] for ki in range(k_actual)]

        for gi, group in enumerate(param_groups):
            with ablated_params(group):
                p_ablated_orig = _get_loss(model, tokenizer, texts)

                # k forward passes of B samples each — same cost as original but 1/k the memory
                p_interv_cols = torch.stack([
                    _get_loss(model, tokenizer, interv_batch)
                    for interv_batch in transposed
                ], dim=1)  # (B, k)

            delta_vals = (p_original.unsqueeze(1) - p_interv_cols).abs().mean(dim=1)
            delta_causal_vals = (p_ablated_orig.unsqueeze(1) - p_interv_cols).abs().mean(dim=1)

            delta_sum[gi]        += delta_vals.mean().item()
            delta_causal_sum[gi] += delta_causal_vals.mean().item()

        n_batches += 1
        torch.cuda.empty_cache()
        if max_batches and n_batches >= max_batches:
            break

    if n_batches == 0:
        logger.warning("No batches processed in Stage I.")
        return [], {}

    delta        = delta_sum        / n_batches
    delta_causal = delta_causal_sum / n_batches

    scores = {}
    theta_s = []
    for gi, group in enumerate(param_groups):
        d  = delta[gi].item()
        dc = delta_causal[gi].item()
        scores[group["name"]] = {"delta": d, "delta_causal": dc}

        if d > tau1 and dc < tau2:
            theta_s.append(group)

    logger.info(
        f"Stage I complete: {len(theta_s)}/{n_groups} groups selected "
        f"(τ1={tau1}, τ2={tau2})."
    )
    return theta_s, scores
