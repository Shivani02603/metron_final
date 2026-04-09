"""
Core configuration: LLM providers, rate limits, token budgets.
Sourced from existing METRON backend (app_v3.py LLM_PROVIDERS dict).
"""

import os
from typing import Dict, Any

# ── LLM Provider Registry ──────────────────────────────────────────────────
LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "NVIDIA NIM": {
        "prefix": "nvidia_nim",
        "models": {
            "fast":     "nvidia_nim/meta/llama-3.1-8b-instruct",
            "judge":    "nvidia_nim/meta/llama-3.1-70b-instruct",
            "balanced": "nvidia_nim/meta/llama-3.1-70b-instruct",
        },
        "default": "nvidia_nim/meta/llama-3.1-70b-instruct",
        "env_key": "NVIDIA_NIM_API_KEY",
        "rpm": 40,
        "description": "Default | 40 RPM Free | Best balance",
        "token_optimize": False,
    },
    "Azure OpenAI": {
        "prefix": "azure",
        "models": {
            "fast":     "azure/gpt-4o",
            "judge":    "azure/gpt-4o",
            "balanced": "azure/gpt-4o",
        },
        "default": "azure/gpt-4o",
        "env_key": "AZURE_OPENAI_API_KEY",
        "endpoint_key": "AZURE_OPENAI_ENDPOINT",
        "rpm": 300,
        "tpm": 49000,
        "description": "Azure GPT-4o | 300 RPM | 49K TPM",
        "token_optimize": True,   # compact prompts — stay within 49K TPM
    },
    "Groq": {
        "prefix": "groq",
        "models": {
            "fast":     "groq/llama-3.1-8b-instant",
            "judge":    "groq/llama-3.3-70b-versatile",
            "balanced": "groq/llama-3.3-70b-versatile",
        },
        "default": "groq/llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "rpm": 30,
        "description": "Very Fast | 30 RPM | 100K tokens/day free",
        "token_optimize": False,
    },
    "Google Gemini": {
        "prefix": "gemini",
        "models": {
            "fast":     "gemini/gemini-2.0-flash",
            "judge":    "gemini/gemini-1.5-pro-latest",
            "balanced": "gemini/gemini-2.0-flash",
        },
        "default": "gemini/gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
        "rpm": 60,
        "description": "Fast | 60 RPM | 1M tokens/day free",
        "token_optimize": False,
    },
}

# Auto-fallback chain when primary provider hits 429 / quota exhaustion
FALLBACK_CHAIN = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
]

# ── Token budgets ──────────────────────────────────────────────────────────
TOKEN_BUDGET_COMPACT = 1200   # Azure: balanced quality within 49K TPM
TOKEN_BUDGET_NORMAL  = 1500   # Free tiers
TOKEN_BUDGET_LARGE   = 4000   # Persona generation (detailed JSON output)

# ── Domain weight profiles for health score ────────────────────────────────
DOMAIN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "finance":    {"functional": 0.35, "security": 0.40, "quality": 0.10, "performance": 0.10, "load": 0.05},
    "banking":    {"functional": 0.35, "security": 0.40, "quality": 0.10, "performance": 0.10, "load": 0.05},
    "medical":    {"functional": 0.35, "security": 0.45, "quality": 0.10, "performance": 0.08, "load": 0.02},
    "healthcare": {"functional": 0.35, "security": 0.45, "quality": 0.10, "performance": 0.08, "load": 0.02},
    "legal":      {"functional": 0.35, "security": 0.40, "quality": 0.15, "performance": 0.07, "load": 0.03},
    "travel":     {"functional": 0.35, "security": 0.15, "quality": 0.10, "performance": 0.25, "load": 0.15},
    "ecommerce":  {"functional": 0.35, "security": 0.20, "quality": 0.10, "performance": 0.20, "load": 0.15},
    "retail":     {"functional": 0.35, "security": 0.20, "quality": 0.10, "performance": 0.20, "load": 0.15},
    # default: balanced
    "_default":   {"functional": 0.40, "security": 0.30, "quality": 0.10, "performance": 0.15, "load": 0.05},
}

HIGH_SECURITY_DOMAINS = {"finance", "banking", "medical", "healthcare", "legal", "government"}
HIGH_TRAFFIC_DOMAINS  = {"travel", "ecommerce", "retail", "booking", "support"}

# ── Evaluation thresholds ──────────────────────────────────────────────────
THRESHOLDS = {
    "health_score_pass":      0.70,
    "functional_pass":        0.50,
    "security_pass":          0.50,
    "quality_pass":           0.70,
    "performance_latency_ms": 5000,
    "hallucination_max":      0.50,  # lower is better (inverted)
    "toxicity_max":           0.30,
    "bias_max":               0.40,
    "rouge_l_min":            0.30,
    "bert_score_min":         0.85,
}

# ── CORS ───────────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# ── Helper: resolve API key from env or explicit value ─────────────────────
def resolve_api_key(provider_name: str, explicit_key: str = "") -> str:
    if explicit_key:
        return explicit_key
    env_key = LLM_PROVIDERS.get(provider_name, {}).get("env_key", "")
    return os.environ.get(env_key, "") if env_key else ""

def get_model(provider_name: str, task: str = "balanced") -> str:
    """Return litellm model string for provider + task (fast/judge/balanced)."""
    p = LLM_PROVIDERS.get(provider_name, LLM_PROVIDERS["Groq"])
    return p["models"].get(task, p["default"])

def should_optimize_tokens(provider_name: str) -> bool:
    return LLM_PROVIDERS.get(provider_name, {}).get("token_optimize", False)

def get_token_budget(provider_name: str, task: str = "normal") -> int:
    if should_optimize_tokens(provider_name):
        return TOKEN_BUDGET_COMPACT
    if task == "large":
        return TOKEN_BUDGET_LARGE
    return TOKEN_BUDGET_NORMAL
