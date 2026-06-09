"""Downloads and loads SPARTQA and StepGame from HuggingFace."""

import logging
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATASETS, SEED, MAX_SEQ_LEN, BATCH_SIZE_TRAIN

logger = logging.getLogger(__name__)


class SpatialQADataset(Dataset):
    """Wraps a HuggingFace dataset split into a PyTorch Dataset."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _format_spartqa(example):
    """Format a SPARTQA example into (input_text, label)."""
    context  = example.get("context", example.get("story", ""))
    question = example.get("question", "")
    # answers field varies; handle list or string
    answers = example.get("answers", example.get("answer", ""))
    if isinstance(answers, list):
        label = answers[0] if answers else ""
    else:
        label = str(answers)
    input_text = f"{context} Question: {question}"
    return {"input_text": input_text.strip(), "label": str(label).strip()}


def _format_stepgame(example):
    """Format a StepGame example into (input_text, label)."""
    story    = example.get("story",    example.get("context", ""))
    question = example.get("question", "")
    label    = example.get("label",    example.get("answer", ""))
    if isinstance(label, list):
        label = label[0] if label else ""
    input_text = f"{story} Question: {question}"
    return {"input_text": input_text.strip(), "label": str(label).strip()}


def _manual_split(dataset, seed=SEED):
    """Fall back to 70/15/15 manual split when official splits are absent."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]
    return (
        dataset.select(train_idx),
        dataset.select(val_idx),
        dataset.select(test_idx),
    )


def load_spatial_dataset(name: str):
    """
    Load a named dataset and return (train_dl, val_dl, test_dl).

    Parameters
    ----------
    name : str
        One of 'spartqa' or 'stepgame'.

    Returns
    -------
    tuple of DataLoader
    """
    hf_name = DATASETS[name]
    formatter = _format_spartqa if name == "spartqa" else _format_stepgame

    logger.info(f"Loading dataset '{name}' from '{hf_name}' …")
    try:
        ds = load_dataset(hf_name)
    except Exception as exc:
        logger.error(f"Failed to download {hf_name}: {exc}")
        raise

    # Identify splits
    available = list(ds.keys())
    logger.info(f"Available splits: {available}")

    if "train" in available and "validation" in available and "test" in available:
        train_raw = ds["train"]
        val_raw   = ds["validation"]
        test_raw  = ds["test"]
    elif "train" in available and "test" in available:
        logger.warning("No validation split found — splitting train 82/18 for val.")
        full = ds["train"]
        split = full.train_test_split(test_size=0.18, seed=SEED)
        train_raw = split["train"]
        val_raw   = split["test"]
        test_raw  = ds["test"]
    else:
        logger.warning("Non-standard splits — applying 70/15/15 manual split.")
        full_key = available[0]
        train_raw, val_raw, test_raw = _manual_split(ds[full_key])

    def _build_examples(raw):
        out = []
        for ex in raw:
            try:
                out.append(formatter(ex))
            except Exception:
                pass
        return out

    train_ex = _build_examples(train_raw)
    val_ex   = _build_examples(val_raw)
    test_ex  = _build_examples(test_raw)

    logger.info(f"  train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}")

    def _collate(batch):
        return {
            "input_text": [b["input_text"] for b in batch],
            "label":      [b["label"]      for b in batch],
        }

    train_dl = DataLoader(SpatialQADataset(train_ex), batch_size=BATCH_SIZE_TRAIN,
                          shuffle=True, collate_fn=_collate)
    val_dl   = DataLoader(SpatialQADataset(val_ex),   batch_size=BATCH_SIZE_TRAIN,
                          shuffle=False, collate_fn=_collate)
    test_dl  = DataLoader(SpatialQADataset(test_ex),  batch_size=BATCH_SIZE_TRAIN,
                          shuffle=False, collate_fn=_collate)

    return train_dl, val_dl, test_dl


def load_geo_dataset(hf_name: str):
    """Load real-world geography dataset, return (train_dl, val_dl, test_dl)."""
    logger.info(f"Loading geo dataset from '{hf_name}' …")
    try:
        ds = load_dataset(hf_name)
    except Exception as exc:
        logger.error(f"Failed to load geo dataset {hf_name}: {exc}")
        raise

    available = list(ds.keys())

    def _format_geo(ex):
        context  = ex.get("context", ex.get("passage", ex.get("text", "")))
        question = ex.get("question", "")
        label    = ex.get("label", ex.get("answer", ex.get("answers", "")))
        if isinstance(label, list):
            label = label[0] if label else ""
        return {"input_text": f"{context} Question: {question}".strip(),
                "label": str(label).strip()}

    if "train" in available and "test" in available:
        train_raw = ds["train"]
        val_raw   = ds.get("validation", ds["test"])
        test_raw  = ds["test"]
    else:
        train_raw, val_raw, test_raw = _manual_split(ds[available[0]])

    def _build(raw):
        out = []
        for ex in raw:
            try:
                out.append(_format_geo(ex))
            except Exception:
                pass
        return out

    def _collate(batch):
        return {"input_text": [b["input_text"] for b in batch],
                "label":      [b["label"]      for b in batch]}

    train_dl = DataLoader(SpatialQADataset(_build(train_raw)),
                          batch_size=BATCH_SIZE_TRAIN, shuffle=True,  collate_fn=_collate)
    val_dl   = DataLoader(SpatialQADataset(_build(val_raw)),
                          batch_size=BATCH_SIZE_TRAIN, shuffle=False, collate_fn=_collate)
    test_dl  = DataLoader(SpatialQADataset(_build(test_raw)),
                          batch_size=BATCH_SIZE_TRAIN, shuffle=False, collate_fn=_collate)
    return train_dl, val_dl, test_dl
