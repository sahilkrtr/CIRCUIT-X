"""Minimal circuit discovery via sigmoid-relaxed binary mask optimisation."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (LAMBDA1, LAMBDA2, LR_STAGE2, EPOCHS_STAGE2, EPSILON,
                    MC_SAMPLES_K, MAX_SEQ_LEN, CHECKPOINT_DIR)
from data.interventions import build_interventions
from models.circuit import CircuitMask, MaskedModel

logger = logging.getLogger(__name__)


def _cross_entropy_loss(masked_model: MaskedModel, tokenizer,
                        texts: List[str]) -> torch.Tensor:
                        
    device = next(masked_model.base_model.parameters()).device
    enc = tokenizer(
        texts,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=True,
        return_attention_mask=True,
    )
    input_ids  = enc["input_ids"].to(device)
    attn_mask  = enc["attention_mask"].to(device)

    out = masked_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = out.logits  # (B, T, V)
    del out              # free CausalLMOutput; logits still alive via local var

    # Extract labels/mask from the already-allocated CPU tensors before any del
    shift_labels = input_ids[:, 1:].contiguous().to(logits.device)
    shift_mask   = attn_mask[:, 1:].contiguous().float().to(logits.device)
    del input_ids, attn_mask   # free small GPU inputs

    shift_logits = logits[:, :-1, :].contiguous()  # new storage (not a view)
    del logits                 # free original logits — shift_logits is independent

    loss_fct     = nn.CrossEntropyLoss(reduction="none")
    per_tok_loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).reshape(shift_labels.size())

    masked_loss = per_tok_loss * shift_mask
    seq_len     = shift_mask.sum(dim=1).clamp(min=1)
    del per_tok_loss, shift_mask, shift_labels
    return (masked_loss.sum(dim=1) / seq_len).mean()


def _eval_accuracy(masked_model: MaskedModel, tokenizer,
                   dataloader: DataLoader, candidate_answers: List[str],
                   max_batches: int = 10) -> float:
    """Quick batched accuracy estimate on the validation loader."""
    from metrics.evaluate import _predict_batch
    masked_model.base_model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            texts  = batch["input_text"]
            labels = batch["label"]
            results = _predict_batch(masked_model, tokenizer, texts, candidate_answers)
            for (pred, _), label in zip(results, labels):
                if str(pred).strip().lower() == str(label).strip().lower():
                    correct += 1
                total += 1
    return correct / max(total, 1)


def discover_circuit(
    model: nn.Module,
    tokenizer,
    theta_s: List[Dict],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    candidate_answers: List[str],
    base_accuracy: float,
    lambda1: float = LAMBDA1,
    lambda2: float = LAMBDA2,
    lr: float = LR_STAGE2,
    epochs: int = EPOCHS_STAGE2,
    epsilon: float = EPSILON,
    k: int = MC_SAMPLES_K,
    max_batches_per_epoch: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[CircuitMask, MaskedModel]:
    """Optimise mask variables over candidate groups to find the minimal causal circuit."""
    if not theta_s:
        logger.warning("Stage II: no candidate groups — returning empty mask.")
        mask   = CircuitMask([])
        mmodel = MaskedModel(model, mask)
        return mask, mmodel

    mask   = CircuitMask(theta_s)
    mmodel = MaskedModel(model, mask)

    # Only mask logits are trainable
    optimizer = optim.Adam(mask.mask_logits.parameters(), lr=lr)

    best_val_acc  = -1.0
    best_state    = {k: v.clone() for k, v in mask.state_dict().items()}
    min_acc_threshold = base_accuracy - epsilon

    logger.info(
        f"Stage II: optimising {len(theta_s)} mask variables for {epochs} epochs "
        f"(λ1={lambda1}, λ2={lambda2}, ε={epsilon:.2%}) …"
    )

    for epoch in range(epochs):
        mask.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in tqdm(train_dataloader, desc=f"Stage II epoch {epoch+1}/{epochs}"):
            texts  = batch["input_text"]
            labels = batch["label"]

            optimizer.zero_grad(set_to_none=True)

            MINI_BS  = 8
            n_total  = len(texts)
            n_acc    = max(1, (n_total + MINI_BS - 1) // MINI_BS)
            batch_loss = 0.0

            for mb_start in range(0, n_total, MINI_BS):
                sub = texts[mb_start: mb_start + MINI_BS]
                weight = len(sub) / n_total   # correct for uneven last chunk

                # Task loss mini-step
                lt = _cross_entropy_loss(mmodel, tokenizer, sub) * weight
                lt.backward()
                batch_loss += lt.item()
                del lt
                torch.cuda.empty_cache()

                # Intervention loss mini-step
                sub_p = [build_interventions(t, k=1)[0] for t in sub]
                li = _cross_entropy_loss(mmodel, tokenizer, sub_p) * (weight * lambda2)
                li.backward()
                batch_loss += li.item()
                del li
                torch.cuda.empty_cache()

            # Sparsity loss (scalar CPU grad, computed once per batch)
            ls = mask.sparsity_loss() * lambda1
            ls.backward()
            batch_loss += ls.item()
            del ls

            optimizer.step()
            epoch_loss += batch_loss
            n_batches  += 1
            if max_batches_per_epoch and n_batches >= max_batches_per_epoch:
                break

        avg_loss = epoch_loss / max(n_batches, 1)
        torch.cuda.empty_cache()

        # Validate (2 batches — enough to gauge accuracy without large memory use)
        mask.eval()
        val_acc = _eval_accuracy(mmodel, tokenizer, val_dataloader, candidate_answers,
                                 max_batches=2)
        torch.cuda.empty_cache()

        logger.info(
            f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f} "
            f"(threshold={min_acc_threshold:.4f})"
        )

        # Early stopping: accuracy must stay above base − ε
        if val_acc >= min_acc_threshold and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in mask.state_dict().items()}

        # If accuracy already collapsed too far, restore and stop
        if val_acc < min_acc_threshold - 0.05:
            logger.info("  Early stopping: accuracy dropped too far.")
            break

    mask.load_state_dict(best_state)

    if checkpoint_path:
        os.makedirs(checkpoint_path, exist_ok=True)
        path = os.path.join(checkpoint_path, "circuit_mask.pt")
        torch.save(best_state, path)
        logger.info(f"  Mask saved to {path}")

    logger.info(
        f"Stage II complete: best val_acc={best_val_acc:.4f}, "
        f"PE={mask.parameter_efficiency(sum(p.numel() for p in model.parameters())):.4f}"
    )
    return mask, mmodel
