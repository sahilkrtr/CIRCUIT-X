"""Loads LLaMA-2-7B, Mistral-7B, Gemma-7B and exposes structured parameter groups."""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import BACKBONE_MODELS, MAX_SEQ_LEN, QUANTIZE_7B, QUANTIZE_70B, NUM_GPUS

logger = logging.getLogger(__name__)

logger.debug(f"Backbone loader: {NUM_GPUS} GPU(s) detected.")


def load_model_and_tokenizer(model_key: str, quantize_4bit: Optional[bool] = None):
    """Load a backbone LLM and its tokenizer.

    Parameters
    ----------
    model_key     : str        — key in BACKBONE_MODELS or raw HuggingFace path
    quantize_4bit : bool|None  — True=force 4-bit, False=force fp16, None=auto-detect
                                 Auto: 7B models use fp16 (fit on 24 GB); larger
                                 models use 4-bit when total VRAM < 80 GB.

    Returns
    -------
    (model, tokenizer, revision_hash)
    """
    hf_name = BACKBONE_MODELS.get(model_key, model_key)

    # Auto-detect quantisation need based on model size and available VRAM
    if quantize_4bit is None:
        param_count_b = _estimate_param_billions(hf_name)
        if param_count_b >= 30:
            quantize_4bit = QUANTIZE_70B   # large model — quantise if < 80 GB total
        else:
            quantize_4bit = QUANTIZE_7B    # 7B model — only quantise if < 16 GB/GPU

    logger.info(f"Loading '{hf_name}' | 4-bit={quantize_4bit} | GPUs={NUM_GPUS}")

    kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",        # automatically spreads across all available GPUs
        trust_remote_code=True,
    )
    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # nested quantisation for extra compression
            bnb_4bit_quant_type="nf4",
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(hf_name, **kwargs)
    except Exception as exc:
        logger.error(f"Could not load model {hf_name}: {exc}")
        raise

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Record the commit hash for reproducibility
    try:
        from huggingface_hub import model_info
        info = model_info(hf_name)
        rev = info.sha
    except Exception:
        rev = "unknown"

    logger.info(f"Loaded {hf_name} (revision={rev}) | "
                f"device_map=auto distributing across {NUM_GPUS} GPU(s)")
    return model, tokenizer, rev


def _estimate_param_billions(hf_name: str) -> float:
    """Estimate model size in billions from its HuggingFace name (heuristic)."""
    name_lower = hf_name.lower()
    for size_str, size_b in [("70b", 70), ("67b", 67), ("72b", 72),
                              ("34b", 34), ("13b", 13), ("7b", 7), ("8b", 8)]:
        if size_str in name_lower:
            return size_b
    return 7   # default assumption


def get_parameter_groups(model: nn.Module) -> List[Dict]:
    """Return all transformer blocks and attention heads as named parameter groups.

    Each group is a dict with keys:
      'name'   : str
      'type'   : 'block' | 'attention_head'
      'params' : List[Tuple[str, nn.Parameter]]
    """
    groups = []

    # Walk named modules and identify transformer blocks / attention layers
    for module_name, module in model.named_modules():
        cls_name = type(module).__name__.lower()

        # Transformer decoder blocks (LlamaDecoderLayer, MistralDecoderLayer, etc.)
        if "decoderlayer" in cls_name or "transformerblock" in cls_name:
            params = list(module.named_parameters(prefix=module_name))
            if params:
                groups.append({"name": module_name, "type": "block", "params": params})

        # Attention sub-modules
        elif any(k in cls_name for k in ("attention", "attn")) and "." in module_name:
            params = list(module.named_parameters(prefix=module_name))
            if params:
                groups.append({"name": module_name, "type": "attention_head",
                                "params": params})

    # De-duplicate by name
    seen, unique = set(), []
    for g in groups:
        if g["name"] not in seen:
            seen.add(g["name"])
            unique.append(g)

    logger.info(f"Found {len(unique)} parameter groups.")
    return unique


def get_answer_probabilities(
    model, tokenizer, input_texts: List[str],
    candidate_answers: List[str], device: str = "cuda"
) -> torch.Tensor:
    """Score each candidate answer and return the probability of the argmax answer."""
    model.eval()
    batch_probs = []

    with torch.no_grad():
        for text in input_texts:
            answer_scores = []
            for ans in candidate_answers:
                prompt = f"{text}\nAnswer: {ans}"
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=MAX_SEQ_LEN,
                    truncation=True,
                    padding=False,
                ).to(model.device)

                out = model(**enc, labels=enc["input_ids"])
                # negative log-likelihood → lower is more probable
                answer_scores.append(-out.loss.item())

            scores_t = torch.tensor(answer_scores, dtype=torch.float32)
            probs    = torch.softmax(scores_t, dim=0)
            batch_probs.append(probs.max().item())

    return torch.tensor(batch_probs)


def predict_answers(
    model, tokenizer, input_texts: List[str],
    candidate_answers: List[str]
) -> List[str]:
    """Return the argmax answer for each input text."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for text in input_texts:
            answer_scores = []
            for ans in candidate_answers:
                prompt = f"{text}\nAnswer: {ans}"
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=MAX_SEQ_LEN,
                    truncation=True,
                    padding=False,
                ).to(model.device)
                out = model(**enc, labels=enc["input_ids"])
                answer_scores.append(-out.loss.item())

            best = candidate_answers[int(torch.tensor(answer_scores).argmax())]
            predictions.append(best)

    return predictions


@contextmanager
def ablated_params(param_group: Dict):
    """Context manager that temporarily zeros a parameter group's weights."""
    saved = []
    for name, param in param_group["params"]:
        saved.append(param.data.clone())
        param.data.zero_()
    try:
        yield
    finally:
        for (name, param), original in zip(param_group["params"], saved):
            param.data.copy_(original)
        del saved
        torch.cuda.empty_cache()
