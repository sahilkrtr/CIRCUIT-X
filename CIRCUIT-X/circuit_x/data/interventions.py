"""Builds perturbed inputs for spatial relation interventions."""

import random
import re

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SPATIAL_RELATIONS, RELATION_REPLACEMENTS, MC_SAMPLES_K, SEED


def _delete_spatial_cues(text: str) -> str:
    """Remove directional/relational keywords while keeping entity names intact."""
    result = text
    for rel in sorted(SPATIAL_RELATIONS, key=len, reverse=True):
        pattern = r'\b' + re.escape(rel) + r'\b'
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    result = re.sub(r' {2,}', ' ', result).strip()
    return result


def _replace_spatial_relation(text: str, rng: random.Random) -> str:
    """Replace one spatial relation with its opposite."""
    found = []
    for rel in SPATIAL_RELATIONS:
        pattern = r'\b' + re.escape(rel) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found.append(rel)
    if not found:
        return text
    chosen = rng.choice(found)
    replacement = RELATION_REPLACEMENTS.get(chosen, chosen)
    pattern = r'\b' + re.escape(chosen) + r'\b'
    result = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
    return result


def _paraphrase_relation(text: str, rng: random.Random) -> str:
    """Rephrase spatial statements while preserving entity identity.

    Simple rule: for "A <rel> B", emit "B is <inverse-rel> A".
    Falls back to relation replacement if no match found.
    """
    # Match patterns like "X is <relation> Y" or "X, <relation> Y"
    pattern = re.compile(
        r'(\w[\w\s]*?)\s+(?:is\s+)?(' +
        '|'.join(re.escape(r) for r in sorted(SPATIAL_RELATIONS, key=len, reverse=True)) +
        r')\s+([\w][\w\s]*?)(?=[.,;]|$)',
        re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        entity_a = match.group(1).strip()
        relation = match.group(2).strip().lower()
        entity_b = match.group(3).strip()
        inverse  = RELATION_REPLACEMENTS.get(relation, relation)
        paraphrased = f"{entity_b} is {inverse} {entity_a}"
        result = text[:match.start()] + paraphrased + text[match.end():]
        return result
    return _replace_spatial_relation(text, rng)


STRATEGIES = ["delete", "replace", "paraphrase"]


def build_interventions(input_text: str, k: int = MC_SAMPLES_K,
                        seed: int = SEED) -> list:
    """Return k perturbed versions of input_text.

    Cycles through the three intervention strategies and adds small
    variation via a seeded RNG so results are reproducible.

    Parameters
    ----------
    input_text : str
    k          : int  — number of perturbed samples (default MC_SAMPLES_K=5)
    seed       : int

    Returns
    -------
    list[str]  — length k
    """
    rng = random.Random(seed)
    perturbed = []
    for i in range(k):
        strategy = STRATEGIES[i % len(STRATEGIES)]
        if strategy == "delete":
            p = _delete_spatial_cues(input_text)
        elif strategy == "replace":
            p = _replace_spatial_relation(input_text, rng)
        else:
            p = _paraphrase_relation(input_text, rng)
        # Guard: if perturbation is identical to original, apply deletion
        if p == input_text:
            p = _delete_spatial_cues(input_text)
        perturbed.append(p)
    return perturbed


def build_batch_interventions(input_texts: list, k: int = MC_SAMPLES_K,
                              seed: int = SEED) -> list:
    """Return list[list[str]]: for each input, k perturbed versions."""
    return [build_interventions(t, k=k, seed=seed + idx)
            for idx, t in enumerate(input_texts)]
