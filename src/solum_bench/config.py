"""Global configuration, paths, and constants."""

from pathlib import Path

# Project root (solum-bench/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default paths
DEFAULT_QUESTION_BANK = PROJECT_ROOT / "question_bank" / "v1"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# Question bank constants
SUPPORTED_CATEGORIES = [
    "factual",
    "reasoning",
    "practical",
    "data_interpretation",
    "regional_adaptation",
    "safety",
    "completeness",
    "grounding",
]

SUPPORTED_SUBCATEGORIES = [
    "soil_science",
    "climate_ecology",
    "crop_science",
    "regen_practices",
    "causal_chains",
    "nutrient_cycling",
    "comparative",
    "farmer_advice",
    "scenario",
    "data_interpretation",
    "regional",
    "safety_critical",
    "systems_thinking",
    "citation",
]

SUPPORTED_QUESTION_TYPES = [
    "mcq",
    "open_ended",
    "short_answer",
    "data_interpretation",
    "scenario",
    "comparative",
    "multi_turn",
    "regional",
]

SUPPORTED_DIFFICULTIES = ["easy", "medium", "hard"]

SUPPORTED_SCORING_METHODS = [
    "exact_match",
    "keyword_rubric",
    "llm_judge",
    "numeric_grounding",
    "safety_check",
    "composite",
]

SUPPORTED_PROVIDERS = [
    "anthropic",
    "openai",
    "google",
    "vllm_local",
    "hf_local",
]

# Solum's 7 test regions
TEST_REGIONS = [
    "ohio",
    "helsinki",
    "tokyo",
    "nairobi",
    "mato_grosso",
    "canterbury",
    "rajasthan",
]

# Evaluation dimensions
EVAL_DIMENSIONS = [
    "factual_accuracy",
    "reasoning_depth",
    "specificity",
    "completeness",
    "regional_adaptation",
    "data_grounding",
    "practical_actionability",
    "safety",
    "systems_thinking",
    "coherence",
    "citation_fidelity",
    "context_retention",
]

# Default generation parameters
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
