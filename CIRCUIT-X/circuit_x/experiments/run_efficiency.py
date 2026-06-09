"""Computational efficiency metrics: FLOPs, latency, memory, throughput, energy."""

import csv
import json
import logging
import os
import random
import time

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (BACKBONE_MODELS, DATASETS, RESULTS_DIR, CHECKPOINT_DIR,
                    SEED, BATCH_SIZE_TRAIN, MAX_SEQ_LEN,
                    TAU1, TAU2, LAMBDA1, LAMBDA2, LR_STAGE2,
                    EPOCHS_STAGE2, EPSILON, BATCH_SIZE_STAGE1)
from data.loader import load_spatial_dataset
from models.backbone import load_model_and_tokenizer, get_parameter_groups
from metrics.evaluate import _infer_candidates
from stages.stage1 import estimate_importance
from stages.stage2 import discover_circuit

logger = logging.getLogger(__name__)

N_LATENCY_RUNS = 100  # warm+measure forward passes for latency


def _measure_flops(model, tokenizer, sample_text: str) -> float:
    """Estimate FLOPs in GFlops via thop.profile."""
    try:
        from thop import profile
        enc = tokenizer(
            sample_text,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            flops, _ = profile(model, inputs=(enc["input_ids"],), verbose=False)
        return flops / 1e9
    except Exception as e:
        logger.warning(f"FLOPs measurement failed: {e}")
        return float("nan")


def _measure_latency_ms(model, tokenizer, texts, n_runs=N_LATENCY_RUNS) -> float:
    """Median latency in ms over n_runs forward passes."""
    model.eval()
    sample = texts[0]
    enc = tokenizer(
        sample,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation=True,
    ).to(model.device)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(**enc)

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**enc)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    return float(np.median(latencies))


def _measure_memory_gb() -> float:
    """Peak GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def _measure_energy_joules(model, tokenizer, text: str,
                            n_runs: int = 20) -> float:
    """Estimate energy per sample using pynvml GPU power draw."""
    try:
        from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetPowerUsage, nvmlShutdown)
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return float("nan")

    enc = tokenizer(
        text, return_tensors="pt",
        max_length=MAX_SEQ_LEN, truncation=True,
    ).to(model.device)

    energies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model(**enc)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            try:
                power_mw = nvmlDeviceGetPowerUsage(handle)
                energies.append((power_mw / 1000.0) * elapsed)
            except Exception:
                break

    try:
        nvmlShutdown()
    except Exception:
        pass

    return float(np.mean(energies)) if energies else float("nan")


def _parameter_usage_pct(model, masked_model=None) -> float:
    """Active parameters / total parameters × 100."""
    total = sum(p.numel() for p in model.parameters())
    if masked_model is None:
        return 100.0
    if hasattr(masked_model, "circuit_mask"):
        cm = masked_model.circuit_mask
        active = sum(
            sum(p.numel() for _, p in grp["params"])
            for gate, grp in zip(cm.hard_gates(), cm.groups)
            if gate == 1.0
        )
        return 100.0 * active / max(total, 1)
    return 100.0


def _set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def run_efficiency(backbone_keys=None, dataset_keys=None,
                   max_stage1_batches=None, max_stage2_batches=None):
    _set_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    backbone_keys = backbone_keys or list(BACKBONE_MODELS.keys())
    dataset_keys  = dataset_keys  or list(DATASETS.keys())
    rows, skipped = [], []

    for bk in backbone_keys:
        try:
            model, tokenizer, rev = load_model_and_tokenizer(bk)
        except Exception as e:
            msg = f"SKIPPED model {bk}: {e}"
            logger.error(msg)
            skipped.append(msg)
            continue

        for dk in dataset_keys:
            logger.info(f"\nEfficiency: backbone={bk}, dataset={dk}")

            try:
                train_dl, val_dl, test_dl = load_spatial_dataset(dk)
            except Exception as e:
                msg = f"SKIPPED dataset {dk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                continue

            sample_batch = next(iter(test_dl))
            sample_texts = sample_batch["input_text"][:4]
            cands        = []
            for b in train_dl:
                cands.extend(b["label"])
            candidate_answers = _infer_candidates(cands)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            pu_base       = _parameter_usage_pct(model)
            flops_base    = _measure_flops(model, tokenizer, sample_texts[0])
            lat_base      = _measure_latency_ms(model, tokenizer, sample_texts)
            mem_base      = _measure_memory_gb()
            thr_base      = (BATCH_SIZE_TRAIN / (lat_base / 1000)) if lat_base > 0 else float("nan")
            energy_base   = _measure_energy_joules(model, tokenizer, sample_texts[0])

            rows.append({
                "backbone": bk, "dataset": dk, "variant": "Base",
                "P(%)":     f"{pu_base:.1f}",
                "FLOPs(G)": f"{flops_base:.1f}",
                "Lat(ms)":  f"{lat_base:.1f}",
                "Mem(GB)":  f"{mem_base:.1f}",
                "Thr(s/s)": f"{thr_base:.1f}",
                "En(J)":    f"{energy_base:.1f}",
            })
            logger.info(f"  Base: P={pu_base:.1f}%, FLOPs={flops_base:.1f}G, "
                        f"Lat={lat_base:.1f}ms, Mem={mem_base:.1f}GB, "
                        f"Thr={thr_base:.1f}, En={energy_base:.2f}J")

            try:
                param_groups = get_parameter_groups(model)
                theta_s, _ = estimate_importance(
                    model, tokenizer, param_groups, train_dl,
                    tau1=TAU1, tau2=TAU2,
                    max_batches=max_stage1_batches or BATCH_SIZE_STAGE1,
                )
                if not theta_s:
                    raise ValueError("Empty θ_s from Stage I")

                base_acc = 0.5  # not critical for efficiency measurement
                circuit_mask, masked_model = discover_circuit(
                    model, tokenizer, theta_s,
                    train_dl, val_dl, candidate_answers,
                    base_accuracy=base_acc,
                    lambda1=LAMBDA1, lambda2=LAMBDA2,
                    lr=LR_STAGE2, epochs=EPOCHS_STAGE2, epsilon=EPSILON,
                    max_batches_per_epoch=max_stage2_batches,
                )
            except Exception as e:
                msg = f"SKIPPED CIRCUIT-X stage {bk}/{dk}: {e}"
                logger.error(msg)
                skipped.append(msg)
                continue

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            pu_cx       = _parameter_usage_pct(model, masked_model)
            flops_cx    = _measure_flops(masked_model, tokenizer, sample_texts[0])
            lat_cx      = _measure_latency_ms(masked_model, tokenizer, sample_texts)
            mem_cx      = _measure_memory_gb()
            thr_cx      = (BATCH_SIZE_TRAIN / (lat_cx / 1000)) if lat_cx > 0 else float("nan")
            energy_cx   = _measure_energy_joules(masked_model, tokenizer, sample_texts[0])

            rows.append({
                "backbone": bk, "dataset": dk, "variant": "CIRCUIT-X",
                "P(%)":     f"{pu_cx:.1f}",
                "FLOPs(G)": f"{flops_cx:.1f}",
                "Lat(ms)":  f"{lat_cx:.1f}",
                "Mem(GB)":  f"{mem_cx:.1f}",
                "Thr(s/s)": f"{thr_cx:.1f}",
                "En(J)":    f"{energy_cx:.1f}",
            })
            logger.info(f"  CIRCUIT-X: P={pu_cx:.1f}%, FLOPs={flops_cx:.1f}G, "
                        f"Lat={lat_cx:.1f}ms, Mem={mem_cx:.1f}GB, "
                        f"Thr={thr_cx:.1f}, En={energy_cx:.2f}J")

        del model

    csv_path  = os.path.join(RESULTS_DIR, "table6.csv")
    json_path = os.path.join(RESULTS_DIR, "table6.json")
    skip_path = os.path.join(RESULTS_DIR, "skipped_experiments.log")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Table 6 saved to {csv_path}")

    if skipped:
        with open(skip_path, "a") as f:
            f.write("\n=== Table 6 ===\n")
            f.writelines(s + "\n" for s in skipped)

    return rows
