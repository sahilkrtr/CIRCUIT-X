import os
import torch

NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
VRAM_PER_GPU_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1e9
    if NUM_GPUS > 0 else 0
)
TOTAL_VRAM_GB = VRAM_PER_GPU_GB * NUM_GPUS

QUANTIZE_7B  = VRAM_PER_GPU_GB < 16
QUANTIZE_70B = TOTAL_VRAM_GB   < 80
CAN_RUN_70B  = TOTAL_VRAM_GB   >= 40

TAU1 = 0.05
TAU2 = 0.02
BATCH_SIZE_STAGE1 = 32
MC_SAMPLES_K = 5

LAMBDA1 = 1e-4        # L1 sparsity coefficient
LAMBDA2 = 1.0         # Intervention robustness coefficient
LR_STAGE2 = 1e-3      # Adam learning rate for mask optimisation
EPOCHS_STAGE2 = 10
EPSILON = 0.02        # Max allowed accuracy degradation

BATCH_SIZE_TRAIN = 32
MAX_SEQ_LEN = 512

GRANULARITY = "block_and_head"

BACKBONE_MODELS = {
    "llama2_7b":  "meta-llama/Llama-2-7b-hf",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma_7b":   "google/gemma-7b-it",
}

DATASETS = {
    "spartqa":  "tasksource/spartqa-mchoice",
    "stepgame": "tasksource/stepgame",
}

GEO_DATASET_HF = "ffaisal93/dataset_geography"

TAU1_SWEEP    = [0.01, 0.03, 0.05, 0.10]
TAU2_SWEEP    = [0.005, 0.01, 0.02, 0.05]
LAMBDA1_SWEEP = [1e-5, 1e-4, 1e-3, 1e-2]
LAMBDA2_SWEEP = [0.1, 0.5, 1.0, 2.0]

ABLATION_VARIANTS = [
    "base",
    "stage1_no_causal_filter",
    "stage1_full",
    "stage2_no_sparsity",
    "stage2_no_intervention",
    "full_circuit_x",
]

RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

SEED = 42

SPATIAL_RELATIONS = [
    "north of", "south of", "east of", "west of",
    "northeast of", "northwest of", "southeast of", "southwest of",
    "left of", "right of", "above", "below",
    "between", "inside", "outside", "near",
    "in front of", "behind", "beside", "next to",
    "on top of", "under", "over",
]

RELATION_REPLACEMENTS = {
    "north of":     "south of",
    "south of":     "north of",
    "east of":      "west of",
    "west of":      "east of",
    "left of":      "right of",
    "right of":     "left of",
    "above":        "below",
    "below":        "above",
    "northeast of": "southwest of",
    "northwest of": "southeast of",
    "southeast of": "northwest of",
    "southwest of": "northeast of",
    "in front of":  "behind",
    "behind":       "in front of",
    "inside":       "outside",
    "outside":      "inside",
    "near":         "far from",
    "beside":       "away from",
    "next to":      "away from",
    "on top of":    "under",
    "under":        "on top of",
    "over":         "under",
    "between":      "apart from",
}
