"""Settings of the recorded experiments: query parameters, consumers, serving, seeds, budgets and task lists."""
from __future__ import annotations

EMBEDDER = "sentence-transformers/all-mpnet-base-v2"

K = 1  # answers per query
M = 1  # returned steps that vote in a policy's experience term
LAMBDA = 1.0  # weight of that term
THETA = {"alfworld": 0.6, "scienceworld": 0.8}
GROUND = True

K_LEVELS = {"llm": (1, 2, 4, 8), "policy": (1, 2, 4, 8, 16)}
COVERAGE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
THETA_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)

RUNS = 3
BOOTSTRAP_RESAMPLES = 10_000
SEEDS = {"bootstrap": 0, "policy_training": 0, "coverage_ordering": "task id"}

APPLICATIONS = {
    "alfworld": {
        "split": "eval_out_of_distribution",
        "max_steps": 40,
        "bindings": "scan",
        "tasks": "data/tasks/alfworld_evaluation.txt",
        "store_tasks": "data/tasks/alfworld_store.txt",
        "calibration_tasks": "data/tasks/alfworld_valid_seen.txt",
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
        "plan_tasks": "data/tasks/scienceworld_plan_training.txt",
        "cross_application_tasks": "data/tasks/scienceworld_cross_application.txt",
    },
}

CONSUMERS = {
    "gemma": {"kind": "llm", "producer": "Gemma-26B", "model": "google/gemma-4-26b-a4b-it", "serving": "openrouter",
              "base_url": "https://openrouter.ai/api/v1", "providers": {"alfworld": ("Cloudflare", "DeepInfra"),
                                                                       "scienceworld": ("Cloudflare",)}},
    "ministral": {"kind": "llm", "producer": "Ministral-8B", "model": "mistralai/ministral-8b-2512", "serving": "bedrock",
                  "bedrock_model_id": "mistral.ministral-3-8b-instruct"},
    "dagger": {"kind": "policy", "producer": "DAgger", "application": "alfworld",
               "checkpoint": "data/models/alfworld_dagger/dagger_admissible_id.pt"},
    "expert": {"kind": "expert", "producer": "expert", "application": "alfworld", "split": "train"},
    "imitation": {"kind": "policy", "producer": "imitation", "application": "scienceworld",
                  "directory": "data/models/scienceworld_imitation"},
}

LLM = {
    "temperature": 0.0,
    "reasoning": False,
    "max_tokens": {"alfworld": 4000, "scienceworld": 1200},
    "timeout_seconds": {"alfworld": 300, "scienceworld": 600},
    "retries": {"alfworld": 4, "scienceworld": 6},
    "action_interface": "static",
    "invalid_action": "environment",
    "api_key_variable": {"openrouter": "OPENROUTER_API_KEY", "bedrock": "AWS credentials of the environment"},
}
