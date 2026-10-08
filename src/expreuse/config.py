"""Settings of the recorded experiments: query parameters, consumers, serving, seeds, budgets and task lists.

The scripts in `expreuse.experiments` read these values; their command lines can override any of them.
"""
from __future__ import annotations

EMBEDDER = "sentence-transformers/all-mpnet-base-v2"  # precomputed goal embeddings, 768 dimensions

# Reuse queries
K = 1  # answers per query (retrieval depth)
M = 1  # returned steps that vote in a policy consumer's experience term
LAMBDA = 1.0  # weight of that term against the policy's own log-probabilities
THETA = {"alfworld": 0.6, "scienceworld": 0.8}  # Cascade's similarity threshold, selected on the calibration lists below
GROUND = True  # within-application answers are grounded; cross-application answers never are

# Ablation levels
K_LEVELS = {"llm": (1, 2, 4, 8), "policy": (1, 2, 4, 8, 16)}
COVERAGE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)  # share of the answer's subtraces withheld
THETA_LEVELS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

# Repetitions and seeds
RUNS = 3  # repetitions of every LLM condition; a trained policy is deterministic and runs once
BOOTSTRAP_RESAMPLES = 10_000
SEEDS = {"alfworld_environment": 42, "bootstrap": 0, "policy_training": 0, "coverage_ordering": "task id"}

APPLICATIONS = {
    "alfworld": {
        "split": "eval_out_of_distribution",  # the 134 valid_unseen games
        "max_steps": 40,
        "tasks": "data/tasks/alfworld_evaluation.txt",
        "store_tasks": "data/tasks/alfworld_store.txt",  # training games the producers' traces come from
        "calibration_tasks": "data/tasks/alfworld_valid_seen.txt",  # theta and the policy's temperature
        "cross_application_tasks": "data/tasks/alfworld_cross_application.txt",
    },
    "scienceworld": {
        "split": "test",
        "simplification": "easy",
        "variations_per_type": 5,
        "max_steps": "per task type: STEP_BUDGET in envs/scienceworld.py",
        "tasks": "data/tasks/scienceworld_evaluation.txt",
        "store_tasks": "data/tasks/scienceworld_store.txt",
        "calibration_tasks": "data/tasks/scienceworld_dev.txt",
        "plan_tasks": "data/tasks/scienceworld_plan_training.txt",  # gold traces the per-type plans come from
        "cross_application_tasks": "data/tasks/scienceworld_cross_application.txt",
    },
}

# Consumers and producers. `producer` is the name their traces are stored under.
CONSUMERS = {
    "gemma": {"kind": "llm", "producer": "Gemma-26B", "model": "google/gemma-4-26b-a4b-it", "serving": "openrouter",
              "base_url": "https://openrouter.ai/api/v1", "providers": {"alfworld": ("Cloudflare", "DeepInfra"),
                                                                       "scienceworld": ("Cloudflare",)}},
    "ministral": {"kind": "llm", "producer": "Ministral-8B", "model": "mistralai/ministral-8b-2512", "serving": "bedrock",
                  "bedrock_model_id": "mistral.ministral-3-8b-instruct"},
    "dagger": {"kind": "policy", "producer": "DAgger", "application": "alfworld",
               "checkpoint": "data/models/alfworld_dagger/dagger_admissible_id.pt"},
    "imitation": {"kind": "policy", "producer": "imitation", "application": "scienceworld",
                  "directory": "data/models/scienceworld_imitation"},
}

# Serving of the language models
LLM = {
    "temperature": 0.0,
    "reasoning": False,  # the model's reasoning mode is switched off where the serving offers one
    "max_tokens": {"alfworld": 4000, "scienceworld": 1200},
    "timeout_seconds": {"alfworld": 300, "scienceworld": 600},  # per request
    "retries": {"alfworld": 4, "scienceworld": 6},
    "action_interface": "static",  # the action forms are listed once; no per-step list of admissible actions
    "invalid_action": "environment",  # an action the environment rejects costs a step and returns its answer
    "api_key_variable": {"openrouter": "OPENROUTER_API_KEY", "bedrock": "AWS credentials of the environment"},
}
