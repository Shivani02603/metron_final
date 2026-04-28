"""
Stage 8: Root Cause Analysis (RCA).

Maps failed evaluation results to probabilistic root causes drawn from the
133-point failure taxonomy (8 categories: AI/Model, RAG, Integration, Security,
Latency, TPM, Database, Operational).

Flow:
  1. Extract signals from metric_results + conversations + perf/load metrics
  2. Parse additional_architecture_notes (keyword-based) for extra flags
  3. Filter taxonomy to architecture-relevant points only
  4. Score each relevant point → probability + affected_count + evidence
  5. Return top-N causes as RCAReport
"""

from __future__ import annotations
from typing import Any, Dict, List, Set, Tuple

from core.models import Conversation, MetricResult, RCAFinding, RCAReport, RunConfig


# ── Signal keys (used throughout taxonomy and signal extractor) ────────────
# Each key represents an observable failure pattern extracted from eval results.

_S = {
    "hallucination_fail",
    "answer_relevancy_fail",
    "usefulness_fail",
    "completeness_fail",
    "consistency_fail",
    "judge_fail",
    "pii_detected",
    "injection_detected",
    "toxicity_fail",
    "bias_fail",
    "attack_resistance_fail",
    "security_fail",
    "rag_faithfulness_fail",
    "rag_recall_fail",
    "error_response",
    "rate_limit_error",
    "timeout_error",
    "high_latency",       # avg > 3000 ms
    "very_high_latency",  # avg > 8000 ms or p95 > 10000 ms
    "load_fail",
    "high_error_rate",    # error_rate > 0.10
    "skipped_metrics",
    "multi_turn_fail",
    "long_prompt_fail",
}


# ── Full 133-point failure taxonomy ────────────────────────────────────────
#
# Each entry:
#   id            – unique identifier (e.g. "C1.1")
#   category_id   – parent category (C1–C8)
#   category      – human-readable category name
#   label         – failure point description
#   signals       – signal keys that are evidence for this point
#   base_rate     – prior probability (how common in practice, 0–1)
#   remediation   – short actionable fix hint
#   requires_*    – architecture flags; point excluded when flag not met

TAXONOMY: List[Dict[str, Any]] = [

    # ── Category 1: AI / Model (22 points) ────────────────────────────────
    {
        "id": "C1.1", "category_id": "C1", "category": "AI / Model",
        "label": "Hallucination & Unfaithful Generation",
        "signals": ["hallucination_fail", "usefulness_fail", "judge_fail"],
        "base_rate": 0.75,
        "remediation": "Add retrieval context, few-shot examples, or output schema validation",
    },
    {
        "id": "C1.2", "category_id": "C1", "category": "AI / Model",
        "label": "Poor Tool / Action Selection",
        "signals": ["usefulness_fail", "judge_fail"],
        "base_rate": 0.45, "requires_multi_agent": True,
        "remediation": "Improve tool descriptions and add few-shot tool-selection examples",
    },
    {
        "id": "C1.3", "category_id": "C1", "category": "AI / Model",
        "label": "Infinite Reasoning Loops",
        "signals": ["consistency_fail", "very_high_latency"],
        "base_rate": 0.30,
        "remediation": "Set explicit termination tokens and max-steps limits",
    },
    {
        "id": "C1.4", "category_id": "C1", "category": "AI / Model",
        "label": "Context Window Overflow",
        "signals": ["completeness_fail", "long_prompt_fail", "consistency_fail"],
        "base_rate": 0.50,
        "remediation": "Implement sliding-window summarisation or reduce conversation history",
    },
    {
        "id": "C1.5", "category_id": "C1", "category": "AI / Model",
        "label": "Instruction Drift",
        "signals": ["consistency_fail", "usefulness_fail", "completeness_fail"],
        "base_rate": 0.55,
        "remediation": "Re-inject system prompt every N turns; use structured instruction format",
    },
    {
        "id": "C1.6", "category_id": "C1", "category": "AI / Model",
        "label": "Model Deprecation / Version Change",
        "signals": ["error_response", "judge_fail"],
        "base_rate": 0.20,
        "remediation": "Pin model version in API calls; monitor provider deprecation notices",
    },
    {
        "id": "C1.7", "category_id": "C1", "category": "AI / Model",
        "label": "Zero-Shot Reasoning Failure",
        "signals": ["answer_relevancy_fail", "judge_fail", "hallucination_fail"],
        "base_rate": 0.60,
        "remediation": "Add chain-of-thought prompting or domain-specific few-shot examples",
    },
    {
        "id": "C1.8", "category_id": "C1", "category": "AI / Model",
        "label": "Model Temperature Too High (>0.7) → Random Tool Selection",
        "signals": ["hallucination_fail", "consistency_fail"],
        "base_rate": 0.40,
        "remediation": "Lower temperature to ≤0.3 for tool-calling and factual tasks",
    },
    {
        "id": "C1.9", "category_id": "C1", "category": "AI / Model",
        "label": "Missing Few-Shot Examples in System Prompt",
        "signals": ["hallucination_fail", "answer_relevancy_fail"],
        "base_rate": 0.55,
        "remediation": "Add 2–3 domain-specific few-shot examples to system prompt",
    },
    {
        "id": "C1.10", "category_id": "C1", "category": "AI / Model",
        "label": "No Explicit Stop / Termination Token → Infinite Generation",
        "signals": ["very_high_latency", "consistency_fail"],
        "base_rate": 0.25,
        "remediation": "Add explicit stop sequences; set max_tokens ceiling",
    },
    {
        "id": "C1.11", "category_id": "C1", "category": "AI / Model",
        "label": "Instruction Overload (>4k Tokens) → Lost-in-Middle",
        "signals": ["completeness_fail", "consistency_fail"],
        "base_rate": 0.45,
        "remediation": "Compress system prompt; move secondary rules to retrieval-augmented guidelines",
    },
    {
        "id": "C1.12", "category_id": "C1", "category": "AI / Model",
        "label": "Underspecified Tool Descriptions → Tool Confusion",
        "signals": ["usefulness_fail", "judge_fail"],
        "base_rate": 0.40, "requires_multi_agent": True,
        "remediation": "Write precise tool descriptions with input/output examples and constraints",
    },
    {
        "id": "C1.13", "category_id": "C1", "category": "AI / Model",
        "label": "No Output Schema Validation → Malformed JSON",
        "signals": ["judge_fail", "error_response"],
        "base_rate": 0.35,
        "remediation": "Enforce structured output (JSON mode) and validate against schema before use",
    },
    {
        "id": "C1.14", "category_id": "C1", "category": "AI / Model",
        "label": "Context Window Fragmentation → Oldest Intent Truncated",
        "signals": ["consistency_fail", "completeness_fail", "multi_turn_fail"],
        "base_rate": 0.45,
        "remediation": "Use rolling summary of older turns; prioritise recent + system context",
    },
    {
        "id": "C1.15", "category_id": "C1", "category": "AI / Model",
        "label": "Model Fine-Tuning Drift Without Notice",
        "signals": ["hallucination_fail", "judge_fail"],
        "base_rate": 0.20,
        "remediation": "Run regression tests on model updates; subscribe to provider changelogs",
    },
    {
        "id": "C1.16", "category_id": "C1", "category": "AI / Model",
        "label": "Zero-Shot Tool Use → Wrong Parameter Types",
        "signals": ["usefulness_fail", "judge_fail"],
        "base_rate": 0.35, "requires_multi_agent": True,
        "remediation": "Provide typed tool schemas and few-shot parameter examples",
    },
    {
        "id": "C1.17", "category_id": "C1", "category": "AI / Model",
        "label": "No Memory of Previous Tool Outputs → Repeated Retrieval",
        "signals": ["consistency_fail", "multi_turn_fail"],
        "base_rate": 0.40, "requires_multi_agent": True,
        "remediation": "Cache tool results in session memory; pass prior outputs in context",
    },
    {
        "id": "C1.18", "category_id": "C1", "category": "AI / Model",
        "label": "Asymmetric Tokenisation → Tool Names Never Match",
        "signals": ["error_response", "judge_fail"],
        "base_rate": 0.15,
        "remediation": "Normalise tool name casing; test with the exact tokeniser the model uses",
    },
    {
        "id": "C1.19", "category_id": "C1", "category": "AI / Model",
        "label": "Hidden System Prompt Injection via RAG",
        "signals": ["injection_detected"],
        "base_rate": 0.40, "requires_rag": True,
        "remediation": "Sanitise RAG corpus; apply output filtering before injecting retrieved context",
    },
    {
        "id": "C1.20", "category_id": "C1", "category": "AI / Model",
        "label": "Model-Specific Refusal Patterns (Claude vs GPT)",
        "signals": ["judge_fail", "usefulness_fail"],
        "base_rate": 0.30,
        "remediation": "Adjust system prompt tone per provider; test on target model specifically",
    },
    {
        "id": "C1.21", "category_id": "C1", "category": "AI / Model",
        "label": "No Confidence Scoring → No Fallback to User",
        "signals": ["hallucination_fail", "judge_fail"],
        "base_rate": 0.45,
        "remediation": "Add self-consistency checks; implement explicit uncertainty expression in prompt",
    },
    {
        "id": "C1.22", "category_id": "C1", "category": "AI / Model",
        "label": "Beam Search / Sampling Bugs → Repetitive Generation",
        "signals": ["consistency_fail", "hallucination_fail"],
        "base_rate": 0.15,
        "remediation": "Set frequency/presence penalty; update to latest stable model SDK version",
    },

    # ── Category 2: RAG-Specific (11 points) ──────────────────────────────
    {
        "id": "C2.1", "category_id": "C2", "category": "RAG-Specific",
        "label": "Retrieval Irrelevance",
        "signals": ["rag_faithfulness_fail", "hallucination_fail"],
        "base_rate": 0.65, "requires_rag": True,
        "remediation": "Tune similarity threshold; switch to hybrid (dense + sparse) search",
    },
    {
        "id": "C2.2", "category_id": "C2", "category": "RAG-Specific",
        "label": "Missing Chunks",
        "signals": ["rag_recall_fail", "completeness_fail"],
        "base_rate": 0.55, "requires_rag": True,
        "remediation": "Reduce chunk size; increase top_k; add parent-document retrieval",
    },
    {
        "id": "C2.3", "category_id": "C2", "category": "RAG-Specific",
        "label": "Index Drift",
        "signals": ["rag_faithfulness_fail", "rag_recall_fail"],
        "base_rate": 0.35, "requires_rag": True,
        "remediation": "Set up automated re-indexing on source document updates",
    },
    {
        "id": "C2.4", "category_id": "C2", "category": "RAG-Specific",
        "label": "Embedding Model Latency",
        "signals": ["high_latency", "rag_faithfulness_fail"],
        "base_rate": 0.40, "requires_rag": True,
        "remediation": "Cache embeddings for repeated queries; use a faster embedding model",
    },
    {
        "id": "C2.5", "category_id": "C2", "category": "RAG-Specific",
        "label": "Reranker Failure",
        "signals": ["rag_recall_fail", "hallucination_fail"],
        "base_rate": 0.35, "requires_rag": True,
        "remediation": "Add fallback to score-based ranking if reranker API fails",
    },
    {
        "id": "C2.6", "category_id": "C2", "category": "RAG-Specific",
        "label": "Context Contamination",
        "signals": ["hallucination_fail", "injection_detected"],
        "base_rate": 0.40, "requires_rag": True,
        "remediation": "Filter retrieved chunks for conflicting or adversarial content before injection",
    },
    {
        "id": "C2.7", "category_id": "C2", "category": "RAG-Specific",
        "label": "Hybrid Search Misconfiguration",
        "signals": ["rag_faithfulness_fail", "rag_recall_fail"],
        "base_rate": 0.30, "requires_rag": True,
        "remediation": "Tune BM25 / dense weight ratio; validate both retrieval paths independently",
    },
    {
        "id": "C2.8", "category_id": "C2", "category": "RAG-Specific",
        "label": "Chunk Size Too Large → Relevant Sentence Diluted",
        "signals": ["hallucination_fail", "rag_faithfulness_fail"],
        "base_rate": 0.50, "requires_rag": True,
        "remediation": "Reduce chunk size to 256–512 tokens; use sentence-level splitting",
    },
    {
        "id": "C2.9", "category_id": "C2", "category": "RAG-Specific",
        "label": "No Reranker → Relevant Chunk Below top_k",
        "signals": ["rag_recall_fail"],
        "base_rate": 0.45, "requires_rag": True,
        "remediation": "Add a cross-encoder reranker (e.g. Cohere Rerank, BGE-Reranker)",
    },
    {
        "id": "C2.10", "category_id": "C2", "category": "RAG-Specific",
        "label": "Embedding Model Not Fine-Tuned for Domain",
        "signals": ["rag_faithfulness_fail", "answer_relevancy_fail"],
        "base_rate": 0.45, "requires_rag": True,
        "remediation": "Fine-tune or switch to a domain-specific embedding model",
    },
    {
        "id": "C2.11", "category_id": "C2", "category": "RAG-Specific",
        "label": "Ground Truth Override Misconfiguration",
        "signals": ["rag_faithfulness_fail"],
        "base_rate": 0.25, "requires_rag": True,
        "remediation": "Verify ground truth mappings; ensure context fields are not null",
    },

    # ── Category 3: Integration & API (22 points) ─────────────────────────
    {
        "id": "C3.1", "category_id": "C3", "category": "Integration & API",
        "label": "API Gateway 429 (Rate Limit)",
        "signals": ["rate_limit_error", "error_response"],
        "base_rate": 0.60,
        "remediation": "Implement per-user token buckets; add Retry-After header parsing",
    },
    {
        "id": "C3.2", "category_id": "C3", "category": "Integration & API",
        "label": "API Gateway 401 / 403 (Auth Failure)",
        "signals": ["error_response"],
        "base_rate": 0.30,
        "remediation": "Rotate and securely store API keys; implement OAuth token refresh",
    },
    {
        "id": "C3.3", "category_id": "C3", "category": "Integration & API",
        "label": "LLM Provider Timeout (504)",
        "signals": ["timeout_error", "very_high_latency", "error_response"],
        "base_rate": 0.50,
        "remediation": "Raise gateway timeout above model P99 latency; add async retry with backoff",
    },
    {
        "id": "C3.4", "category_id": "C3", "category": "Integration & API",
        "label": "Retry Storm (Thundering Herd)",
        "signals": ["rate_limit_error", "load_fail", "high_error_rate"],
        "base_rate": 0.45,
        "remediation": "Add exponential backoff with jitter; implement client-side circuit breaker",
    },
    {
        "id": "C3.5", "category_id": "C3", "category": "Integration & API",
        "label": "Tool API Schema Mismatch",
        "signals": ["error_response", "judge_fail"],
        "base_rate": 0.35, "requires_multi_agent": True,
        "remediation": "Version-lock tool API schemas; add schema validation on tool response",
    },
    {
        "id": "C3.6", "category_id": "C3", "category": "Integration & API",
        "label": "Message Queue Backpressure",
        "signals": ["load_fail", "high_error_rate", "very_high_latency"],
        "base_rate": 0.30, "requires_message_queue": True,
        "remediation": "Set queue depth alarms; implement dead-letter queue with exponential backoff",
    },
    {
        "id": "C3.7", "category_id": "C3", "category": "Integration & API",
        "label": "Circuit Breaker Trip",
        "signals": ["error_response", "high_error_rate"],
        "base_rate": 0.35,
        "remediation": "Tune circuit breaker thresholds; add graceful degradation fallback",
    },
    {
        "id": "C3.8", "category_id": "C3", "category": "Integration & API",
        "label": "No Per-User Token Bucket → Single Tenant Exhausts TPM",
        "signals": ["rate_limit_error", "load_fail"],
        "base_rate": 0.40,
        "remediation": "Implement per-user/session TPM quotas with token bucket algorithm",
    },
    {
        "id": "C3.9", "category_id": "C3", "category": "Integration & API",
        "label": "Missing Retry With Jitter → Thundering Herd",
        "signals": ["rate_limit_error", "load_fail", "high_error_rate"],
        "base_rate": 0.45,
        "remediation": "Replace fixed-interval retry with exponential backoff + random jitter",
    },
    {
        "id": "C3.10", "category_id": "C3", "category": "Integration & API",
        "label": "API Gateway Timeout < Model P99 Latency → False 504",
        "signals": ["timeout_error", "error_response"],
        "base_rate": 0.40,
        "remediation": "Set gateway timeout to at least 2× model P99 latency",
    },
    {
        "id": "C3.11", "category_id": "C3", "category": "Integration & API",
        "label": "Embedding Model TPM Not Accounted → Double Consumption",
        "signals": ["rate_limit_error"],
        "base_rate": 0.35, "requires_rag": True,
        "remediation": "Track embedding and LLM TPM separately; set combined quota guard",
    },
    {
        "id": "C3.12", "category_id": "C3", "category": "Integration & API",
        "label": "No Circuit Breaker for Downstream Tool",
        "signals": ["error_response", "high_error_rate"],
        "base_rate": 0.35, "requires_multi_agent": True,
        "remediation": "Wrap each tool call in a circuit breaker with fallback response",
    },
    {
        "id": "C3.13", "category_id": "C3", "category": "Integration & API",
        "label": "LLM Provider Regional Capacity Exhaustion",
        "signals": ["timeout_error", "error_response"],
        "base_rate": 0.25,
        "remediation": "Configure multi-region failover; use provider load-balancing endpoints",
    },
    {
        "id": "C3.14", "category_id": "C3", "category": "Integration & API",
        "label": "Hardcoded API Keys → 401 After Key Rotation",
        "signals": ["error_response"],
        "base_rate": 0.25,
        "remediation": "Store keys in secrets manager; implement zero-downtime key rotation",
    },
    {
        "id": "C3.15", "category_id": "C3", "category": "Integration & API",
        "label": "Message Queue Max Receives Exceeded → Poison DLQ Missing",
        "signals": ["load_fail", "high_error_rate"],
        "base_rate": 0.20, "requires_message_queue": True,
        "remediation": "Configure DLQ on all queues; set max-receive-count alarm",
    },
    {
        "id": "C3.16", "category_id": "C3", "category": "Integration & API",
        "label": "Reranker API Rate Limit Separate from LLM",
        "signals": ["rate_limit_error"],
        "base_rate": 0.30, "requires_rag": True,
        "remediation": "Track reranker API quota independently; add fallback to score-based ranking",
    },
    {
        "id": "C3.17", "category_id": "C3", "category": "Integration & API",
        "label": "No Fallback to Cache on 429",
        "signals": ["rate_limit_error", "error_response"],
        "base_rate": 0.40,
        "remediation": "Add semantic cache layer (e.g. Redis + cosine similarity) as 429 fallback",
    },
    {
        "id": "C3.18", "category_id": "C3", "category": "Integration & API",
        "label": "Webhook Tool Call Synchronous → Holds Thread",
        "signals": ["very_high_latency", "high_latency"],
        "base_rate": 0.30, "requires_multi_agent": True,
        "remediation": "Make webhook tool calls async; use callback pattern or polling",
    },
    {
        "id": "C3.19", "category_id": "C3", "category": "Integration & API",
        "label": "API Version Negotiation Missing (v1 vs v2)",
        "signals": ["error_response"],
        "base_rate": 0.20,
        "remediation": "Pin API version in all HTTP calls; test against both versions during upgrades",
    },
    {
        "id": "C3.20", "category_id": "C3", "category": "Integration & API",
        "label": "Cross-Region Latency > Gateway Timeout",
        "signals": ["high_latency", "very_high_latency"],
        "base_rate": 0.25,
        "remediation": "Deploy in same region as LLM provider; use latency-based routing",
    },
    {
        "id": "C3.21", "category_id": "C3", "category": "Integration & API",
        "label": "OAuth Token Refresh Race → Invalidation",
        "signals": ["error_response"],
        "base_rate": 0.20,
        "remediation": "Use token refresh lock; refresh proactively before expiry",
    },
    {
        "id": "C3.22", "category_id": "C3", "category": "Integration & API",
        "label": "No 429 Retry-After Header Parsing",
        "signals": ["rate_limit_error"],
        "base_rate": 0.35,
        "remediation": "Parse Retry-After header and sleep the exact advised interval",
    },

    # ── Category 4: Security (22 points) ──────────────────────────────────
    {
        "id": "C4.1", "category_id": "C4", "category": "Security",
        "label": "Prompt Injection (Direct)",
        "signals": ["injection_detected", "attack_resistance_fail"],
        "base_rate": 0.60,
        "remediation": "Sanitise user input; use delimiter-based prompt construction; add injection classifier",
    },
    {
        "id": "C4.2", "category_id": "C4", "category": "Security",
        "label": "Indirect Prompt Injection via Retrieved Documents",
        "signals": ["injection_detected", "attack_resistance_fail"],
        "base_rate": 0.50, "requires_rag": True,
        "remediation": "Sanitise RAG corpus on ingest; add instruction-following boundary in prompt template",
    },
    {
        "id": "C4.3", "category_id": "C4", "category": "Security",
        "label": "Data Exfiltration via Tool Calls",
        "signals": ["pii_detected", "attack_resistance_fail"],
        "base_rate": 0.40, "requires_multi_agent": True,
        "remediation": "Apply least-privilege IAM; validate tool output for PII before returning to user",
    },
    {
        "id": "C4.4", "category_id": "C4", "category": "Security",
        "label": "Session Hijacking",
        "signals": ["security_fail"],
        "base_rate": 0.30, "requires_session_db": True,
        "remediation": "Use cryptographically random session IDs; enforce session expiry and rotation",
    },
    {
        "id": "C4.5", "category_id": "C4", "category": "Security",
        "label": "Tool Abuse / Quota Exhaustion",
        "signals": ["rate_limit_error", "attack_resistance_fail"],
        "base_rate": 0.40,
        "remediation": "Add per-session tool call rate limits; require confirmation for destructive actions",
    },
    {
        "id": "C4.6", "category_id": "C4", "category": "Security",
        "label": "Embedding Model Poisoning",
        "signals": ["rag_faithfulness_fail", "security_fail"],
        "base_rate": 0.25, "requires_rag": True,
        "remediation": "Authenticate vector DB writes; audit corpus on ingest",
    },
    {
        "id": "C4.7", "category_id": "C4", "category": "Security",
        "label": "Secrets in Logs",
        "signals": ["pii_detected", "security_fail"],
        "base_rate": 0.45,
        "remediation": "Add PII/secret scrubbing middleware; use structured logging with field allowlists",
    },
    {
        "id": "C4.8", "category_id": "C4", "category": "Security",
        "label": "RAG Corpus Not Sanitised → Indirect Injection",
        "signals": ["injection_detected", "rag_faithfulness_fail"],
        "base_rate": 0.45, "requires_rag": True,
        "remediation": "Run injection-pattern scan on all ingested documents; reject flagged content",
    },
    {
        "id": "C4.9", "category_id": "C4", "category": "Security",
        "label": "Agent Tool Permissions Over-Broad (Same IAM Role for Read and Delete)",
        "signals": ["security_fail", "attack_resistance_fail"],
        "base_rate": 0.40, "requires_multi_agent": True,
        "remediation": "Apply least-privilege IAM per tool; separate read/write roles",
    },
    {
        "id": "C4.10", "category_id": "C4", "category": "Security",
        "label": "No Output Sanitisation → Leaks Internal Traces",
        "signals": ["pii_detected", "security_fail"],
        "base_rate": 0.45,
        "remediation": "Strip stack traces and internal IDs from API responses; add output filter",
    },
    {
        "id": "C4.11", "category_id": "C4", "category": "Security",
        "label": "Session DB Without Encryption",
        "signals": ["security_fail"],
        "base_rate": 0.30, "requires_session_db": True,
        "remediation": "Enable encryption-at-rest and TLS-in-transit for session DB",
    },
    {
        "id": "C4.12", "category_id": "C4", "category": "Security",
        "label": "Missing Cross-Session Isolation",
        "signals": ["consistency_fail", "security_fail"],
        "base_rate": 0.35, "requires_session_db": True,
        "remediation": "Enforce row-level security; namespace all session keys by user_id",
    },
    {
        "id": "C4.13", "category_id": "C4", "category": "Security",
        "label": "Prompt Injection via Tool Output",
        "signals": ["injection_detected"],
        "base_rate": 0.40, "requires_multi_agent": True,
        "remediation": "Treat all tool outputs as untrusted; sanitise before inserting into LLM context",
    },
    {
        "id": "C4.14", "category_id": "C4", "category": "Security",
        "label": "No Rate Limiting on Tool Calls (Email Bomb etc.)",
        "signals": ["rate_limit_error", "attack_resistance_fail"],
        "base_rate": 0.35,
        "remediation": "Add per-user per-tool rate limits; require confirmation for high-impact tools",
    },
    {
        "id": "C4.15", "category_id": "C4", "category": "Security",
        "label": "Vector DB Without Authentication → Poisoning",
        "signals": ["security_fail"],
        "base_rate": 0.30, "requires_vector_db": True,
        "remediation": "Enable vector DB auth; restrict write access to pipeline service account only",
    },
    {
        "id": "C4.16", "category_id": "C4", "category": "Security",
        "label": "Agent Logs Stored in Plaintext",
        "signals": ["pii_detected"],
        "base_rate": 0.40,
        "remediation": "Encrypt logs at rest; mask PII fields before writing to log store",
    },
    {
        "id": "C4.17", "category_id": "C4", "category": "Security",
        "label": "No Input Validation for Tool Arguments (Path Traversal)",
        "signals": ["injection_detected", "attack_resistance_fail"],
        "base_rate": 0.35, "requires_multi_agent": True,
        "remediation": "Validate and sanitise all tool argument values; use allowlist for file paths",
    },
    {
        "id": "C4.18", "category_id": "C4", "category": "Security",
        "label": "Agent Can Read System Environment Variables",
        "signals": ["security_fail", "attack_resistance_fail"],
        "base_rate": 0.30, "requires_multi_agent": True,
        "remediation": "Run agent in sandboxed environment; block env-var access via policy",
    },
    {
        "id": "C4.19", "category_id": "C4", "category": "Security",
        "label": "Session Replay Attack (Predictable Session ID)",
        "signals": ["security_fail"],
        "base_rate": 0.25, "requires_session_db": True,
        "remediation": "Use UUID v4 or CSPRNG for session IDs; rotate on privilege escalation",
    },
    {
        "id": "C4.20", "category_id": "C4", "category": "Security",
        "label": "Missing Audit Trail for Tool Calls",
        "signals": ["security_fail"],
        "base_rate": 0.35,
        "remediation": "Log every tool invocation with user_id, timestamp, args, and result hash",
    },
    {
        "id": "C4.21", "category_id": "C4", "category": "Security",
        "label": "Model Jailbreak via Encoded Prompts (Base64, ROT13)",
        "signals": ["attack_resistance_fail", "injection_detected"],
        "base_rate": 0.40,
        "remediation": "Add encoding-detection layer (base64, hex, rot13 decoder) before LLM input",
    },
    {
        "id": "C4.22", "category_id": "C4", "category": "Security",
        "label": "Embedding Model Inversion → Privacy Leak",
        "signals": ["pii_detected"],
        "base_rate": 0.20, "requires_rag": True,
        "remediation": "Do not embed raw PII; anonymise documents before indexing",
    },

    # ── Category 5: Latency & Performance (22 points) ─────────────────────
    {
        "id": "C5.1", "category_id": "C5", "category": "Latency & Performance",
        "label": "Cold Start (Serverless)",
        "signals": ["very_high_latency"],
        "base_rate": 0.55, "requires_serverless": True,
        "remediation": "Use provisioned concurrency; implement keep-warm ping every 5 min",
    },
    {
        "id": "C5.2", "category_id": "C5", "category": "Latency & Performance",
        "label": "Sequential Tool Calls Without Parallelism",
        "signals": ["high_latency", "very_high_latency"],
        "base_rate": 0.50, "requires_multi_agent": True,
        "remediation": "Build a tool-call DAG; execute independent tools in parallel with asyncio.gather",
    },
    {
        "id": "C5.3", "category_id": "C5", "category": "Latency & Performance",
        "label": "Token Generation Speed",
        "signals": ["high_latency"],
        "base_rate": 0.45,
        "remediation": "Use streaming responses; switch to a faster model for non-critical tasks",
    },
    {
        "id": "C5.4", "category_id": "C5", "category": "Latency & Performance",
        "label": "Vector DB Query Latency",
        "signals": ["high_latency"],
        "base_rate": 0.40, "requires_vector_db": True,
        "remediation": "Load index into memory (HNSW); cache frequent query embeddings",
    },
    {
        "id": "C5.5", "category_id": "C5", "category": "Latency & Performance",
        "label": "Session DB Read / Write Latency",
        "signals": ["high_latency"],
        "base_rate": 0.35, "requires_session_db": True,
        "remediation": "Use in-memory session cache (Redis); write session asynchronously",
    },
    {
        "id": "C5.6", "category_id": "C5", "category": "Latency & Performance",
        "label": "Tool API Network Jitter",
        "signals": ["high_latency"],
        "base_rate": 0.35, "requires_multi_agent": True,
        "remediation": "Add timeout + retry for each tool call; consider co-locating tool APIs",
    },
    {
        "id": "C5.7", "category_id": "C5", "category": "Latency & Performance",
        "label": "Streaming Interruption",
        "signals": ["timeout_error", "very_high_latency"],
        "base_rate": 0.30,
        "remediation": "Add stream reconnect logic; buffer partial responses for resumption",
    },
    {
        "id": "C5.8", "category_id": "C5", "category": "Latency & Performance",
        "label": "Serverless Cold Start for LLM Container (5–15 s)",
        "signals": ["very_high_latency"],
        "base_rate": 0.50, "requires_serverless": True,
        "remediation": "Use provisioned concurrency; pre-warm container on schedule",
    },
    {
        "id": "C5.9", "category_id": "C5", "category": "Latency & Performance",
        "label": "Sequential Tool Calls Without Batching",
        "signals": ["high_latency", "very_high_latency"],
        "base_rate": 0.45, "requires_multi_agent": True,
        "remediation": "Identify independent tool calls; batch them in a single async gather",
    },
    {
        "id": "C5.10", "category_id": "C5", "category": "Latency & Performance",
        "label": "Large RAG Chunk Size (1024 tokens) → Slow Retrieval",
        "signals": ["high_latency"],
        "base_rate": 0.40, "requires_rag": True,
        "remediation": "Reduce chunk size to 256–512 tokens; use hierarchical chunking",
    },
    {
        "id": "C5.11", "category_id": "C5", "category": "Latency & Performance",
        "label": "Synchronous Session DB Write Before Every LLM Call",
        "signals": ["high_latency"],
        "base_rate": 0.35, "requires_session_db": True,
        "remediation": "Write session state asynchronously after response is sent",
    },
    {
        "id": "C5.12", "category_id": "C5", "category": "Latency & Performance",
        "label": "No Streaming → User Waits for Full Generation",
        "signals": ["very_high_latency"],
        "base_rate": 0.45,
        "remediation": "Enable streaming (SSE / WebSocket); stream tokens as they are generated",
    },
    {
        "id": "C5.13", "category_id": "C5", "category": "Latency & Performance",
        "label": "Vector DB Index Not in Memory (SSD Hit)",
        "signals": ["high_latency"],
        "base_rate": 0.35, "requires_vector_db": True,
        "remediation": "Use in-memory HNSW index; allocate sufficient RAM for index size",
    },
    {
        "id": "C5.14", "category_id": "C5", "category": "Latency & Performance",
        "label": "Agent Re-Embeds Same Query Multiple Times",
        "signals": ["high_latency"],
        "base_rate": 0.40, "requires_rag": True,
        "remediation": "Cache query embeddings per session; deduplicate retrieval calls",
    },
    {
        "id": "C5.15", "category_id": "C5", "category": "Latency & Performance",
        "label": "Full Conversation History Sent Each Turn → Token Bloat",
        "signals": ["high_latency", "multi_turn_fail"],
        "base_rate": 0.50,
        "remediation": "Summarise old turns; send only recent N turns + rolling summary",
    },
    {
        "id": "C5.16", "category_id": "C5", "category": "Latency & Performance",
        "label": "No Speculative Execution (Parallel Tool DAG)",
        "signals": ["high_latency"],
        "base_rate": 0.35, "requires_multi_agent": True,
        "remediation": "Implement speculative / parallel tool execution where tasks are independent",
    },
    {
        "id": "C5.17", "category_id": "C5", "category": "Latency & Performance",
        "label": "Cross-AZ Network Hop",
        "signals": ["high_latency"],
        "base_rate": 0.25,
        "remediation": "Pin services to same AZ; use VPC endpoints to avoid internet routing",
    },
    {
        "id": "C5.18", "category_id": "C5", "category": "Latency & Performance",
        "label": "LLM Provider Global Load Balancer → Distant Region",
        "signals": ["high_latency", "very_high_latency"],
        "base_rate": 0.30,
        "remediation": "Specify region-pinned endpoint; use provider's regional API URL",
    },
    {
        "id": "C5.19", "category_id": "C5", "category": "Latency & Performance",
        "label": "Heavy Reranker (BERT-Large) on Top-20 Chunks",
        "signals": ["high_latency"],
        "base_rate": 0.35, "requires_rag": True,
        "remediation": "Use a lighter reranker (MiniLM); reduce reranker candidate count",
    },
    {
        "id": "C5.20", "category_id": "C5", "category": "Latency & Performance",
        "label": "No Response Compression (Large JSON Tool Outputs)",
        "signals": ["high_latency"],
        "base_rate": 0.25,
        "remediation": "Enable gzip/brotli compression on API responses; truncate verbose tool output",
    },
    {
        "id": "C5.21", "category_id": "C5", "category": "Latency & Performance",
        "label": "Garbage Collection in Serverless Runtime",
        "signals": ["very_high_latency"],
        "base_rate": 0.20, "requires_serverless": True,
        "remediation": "Increase memory allocation; use compiled runtime (Rust/Go) for hot paths",
    },
    {
        "id": "C5.22", "category_id": "C5", "category": "Latency & Performance",
        "label": "Synchronous Logging to Cloud Storage (CloudWatch Flush)",
        "signals": ["high_latency"],
        "base_rate": 0.25,
        "remediation": "Buffer logs in memory; flush asynchronously; use structured log batching",
    },

    # ── Category 6: TPM / Rate Limiting (5 points) ────────────────────────
    {
        "id": "C6.1", "category_id": "C6", "category": "TPM / Rate Limiting",
        "label": "Burst Exhaustion",
        "signals": ["rate_limit_error", "load_fail"],
        "base_rate": 0.55,
        "remediation": "Spread requests over time; request TPM quota increase from provider",
    },
    {
        "id": "C6.2", "category_id": "C6", "category": "TPM / Rate Limiting",
        "label": "No Per-User Quota",
        "signals": ["rate_limit_error", "error_response"],
        "base_rate": 0.50,
        "remediation": "Implement per-user token budgets; queue requests when budget nears limit",
    },
    {
        "id": "C6.3", "category_id": "C6", "category": "TPM / Rate Limiting",
        "label": "Retry with Backoff Flood",
        "signals": ["rate_limit_error", "load_fail", "high_error_rate"],
        "base_rate": 0.45,
        "remediation": "Add full jitter to backoff; track global retry count to avoid stampede",
    },
    {
        "id": "C6.4", "category_id": "C6", "category": "TPM / Rate Limiting",
        "label": "Embedding + LLM Combined TPM Overflow",
        "signals": ["rate_limit_error"],
        "base_rate": 0.40, "requires_rag": True,
        "remediation": "Count embedding tokens toward total TPM budget; throttle both independently",
    },
    {
        "id": "C6.5", "category_id": "C6", "category": "TPM / Rate Limiting",
        "label": "Fallback Model Also Hitting 429",
        "signals": ["rate_limit_error", "error_response"],
        "base_rate": 0.30,
        "remediation": "Use different provider or model family as fallback; cache responses for hot queries",
    },

    # ── Category 7: Database & Session (15 points) ────────────────────────
    {
        "id": "C7.1", "category_id": "C7", "category": "Database & Session",
        "label": "Session State Corruption",
        "signals": ["consistency_fail", "error_response"],
        "base_rate": 0.40, "requires_session_db": True,
        "remediation": "Add optimistic locking; validate session schema on every read",
    },
    {
        "id": "C7.2", "category_id": "C7", "category": "Database & Session",
        "label": "Connection Pool Exhaustion",
        "signals": ["load_fail", "high_error_rate", "error_response"],
        "base_rate": 0.45, "requires_session_db": True,
        "remediation": "Increase pool size; add connection wait timeout; use async DB drivers",
    },
    {
        "id": "C7.3", "category_id": "C7", "category": "Database & Session",
        "label": "Row Lock Contention",
        "signals": ["high_latency", "load_fail"],
        "base_rate": 0.30, "requires_session_db": True,
        "remediation": "Use row-level locking; switch to append-only session log pattern",
    },
    {
        "id": "C7.4", "category_id": "C7", "category": "Database & Session",
        "label": "TTL Expiration Mid-Conversation",
        "signals": ["consistency_fail", "multi_turn_fail", "error_response"],
        "base_rate": 0.40, "requires_session_db": True,
        "remediation": "Extend TTL on each turn; use sliding TTL tied to user activity",
    },
    {
        "id": "C7.5", "category_id": "C7", "category": "Database & Session",
        "label": "Serialisation Bloat",
        "signals": ["high_latency"],
        "base_rate": 0.30, "requires_session_db": True,
        "remediation": "Use compact serialisation (MessagePack / protobuf); prune stale session fields",
    },
    {
        "id": "C7.6", "category_id": "C7", "category": "Database & Session",
        "label": "Read-After-Write Inconsistency",
        "signals": ["consistency_fail", "multi_turn_fail"],
        "base_rate": 0.35, "requires_session_db": True,
        "remediation": "Enforce read-your-writes consistency; avoid read-replica for session reads",
    },
    {
        "id": "C7.7", "category_id": "C7", "category": "Database & Session",
        "label": "Session DB Connection Pool Overflow",
        "signals": ["load_fail", "high_error_rate"],
        "base_rate": 0.40, "requires_session_db": True,
        "remediation": "Tune pool max size; add connection queue with bounded wait",
    },
    {
        "id": "C7.8", "category_id": "C7", "category": "Database & Session",
        "label": "No Row-Level Locking → Lost Update",
        "signals": ["consistency_fail"],
        "base_rate": 0.30, "requires_session_db": True,
        "remediation": "Use SELECT FOR UPDATE; implement optimistic concurrency with version column",
    },
    {
        "id": "C7.9", "category_id": "C7", "category": "Database & Session",
        "label": "TTL Too Short → Session Deleted While User Is Thinking",
        "signals": ["consistency_fail", "multi_turn_fail", "error_response"],
        "base_rate": 0.40, "requires_session_db": True,
        "remediation": "Set TTL ≥ 30 min idle; slide TTL on every user action",
    },
    {
        "id": "C7.10", "category_id": "C7", "category": "Database & Session",
        "label": "Read Replica Lag → Inconsistent Memory",
        "signals": ["consistency_fail"],
        "base_rate": 0.30, "requires_session_db": True,
        "remediation": "Route session reads to primary; use replication lag alarm",
    },
    {
        "id": "C7.11", "category_id": "C7", "category": "Database & Session",
        "label": "No Cost Budget Per Session → $50 Runaway Loop",
        "signals": ["rate_limit_error"],
        "base_rate": 0.35, "requires_session_db": True,
        "remediation": "Track token spend per session; hard-stop when budget exceeded",
    },
    {
        "id": "C7.12", "category_id": "C7", "category": "Database & Session",
        "label": "Missing Trace IDs → Cannot Correlate API → Agent Step → DB",
        "signals": ["skipped_metrics"],
        "base_rate": 0.40, "requires_session_db": True,
        "remediation": "Propagate trace_id (e.g. OpenTelemetry) from API → agent → DB calls",
    },
    {
        "id": "C7.13", "category_id": "C7", "category": "Database & Session",
        "label": "Agent Checkpoint Not Stored → Timeout Loses Progress",
        "signals": ["error_response", "consistency_fail"],
        "base_rate": 0.35, "requires_session_db": True,
        "remediation": "Checkpoint agent state after each tool call; resume from last checkpoint on retry",
    },
    {
        "id": "C7.14", "category_id": "C7", "category": "Database & Session",
        "label": "Version Skew Between Agent Code and Stored Session Schema",
        "signals": ["error_response"],
        "base_rate": 0.25, "requires_session_db": True,
        "remediation": "Add schema version field to sessions; run migration on version mismatch",
    },
    {
        "id": "C7.15", "category_id": "C7", "category": "Database & Session",
        "label": "No Dead-Letter Queue for Failed Agent Steps",
        "signals": ["error_response", "load_fail"],
        "base_rate": 0.30, "requires_session_db": True,
        "remediation": "Add DLQ for failed agent tasks; implement retry policy with backoff",
    },

    # ── Category 8: Operational (14 points) ───────────────────────────────
    {
        "id": "C8.1", "category_id": "C8", "category": "Operational",
        "label": "Observability Gap",
        "signals": ["skipped_metrics", "high_error_rate"],
        "base_rate": 0.50,
        "remediation": "Add distributed tracing (OpenTelemetry); set sampling rate to 100% for errors",
    },
    {
        "id": "C8.2", "category_id": "C8", "category": "Operational",
        "label": "Cost Explosion",
        "signals": ["rate_limit_error", "very_high_latency"],
        "base_rate": 0.35,
        "remediation": "Set per-session token budget; add cost anomaly alert (e.g. AWS Cost Anomaly)",
    },
    {
        "id": "C8.3", "category_id": "C8", "category": "Operational",
        "label": "Version Mismatch (Agent Code vs Deployed Model)",
        "signals": ["error_response", "judge_fail"],
        "base_rate": 0.30,
        "remediation": "Pin model version in deployment config; run smoke test on every deploy",
    },
    {
        "id": "C8.4", "category_id": "C8", "category": "Operational",
        "label": "No Graceful Degradation",
        "signals": ["error_response", "high_error_rate"],
        "base_rate": 0.50,
        "remediation": "Implement fallback responses for each failure mode; never surface raw errors to users",
    },
    {
        "id": "C8.5", "category_id": "C8", "category": "Operational",
        "label": "Checkpointing Failure",
        "signals": ["error_response", "consistency_fail"],
        "base_rate": 0.30,
        "remediation": "Test checkpoint restore path; add integration test for mid-run recovery",
    },
    {
        "id": "C8.6", "category_id": "C8", "category": "Operational",
        "label": "Multi-Region Drift",
        "signals": ["high_latency", "consistency_fail"],
        "base_rate": 0.20,
        "remediation": "Use global consistency model; synchronise config/model version across regions",
    },
    {
        "id": "C8.7", "category_id": "C8", "category": "Operational",
        "label": "Testing Blindness (No Adversarial / Edge-Case Coverage)",
        "signals": ["hallucination_fail", "security_fail"],
        "base_rate": 0.55,
        "remediation": "Add adversarial test suite; use red-teaming tools; run Metron regularly",
    },
    {
        "id": "C8.8", "category_id": "C8", "category": "Operational",
        "label": "No Dead-Letter Queue for Failed Agent Steps",
        "signals": ["error_response", "load_fail"],
        "base_rate": 0.30,
        "remediation": "Configure DLQ on all agent task queues; alert on DLQ depth > 0",
    },
    {
        "id": "C8.9", "category_id": "C8", "category": "Operational",
        "label": "Observability Sampling Rate 0.1% → Misses 99.9% of 429 Errors",
        "signals": ["skipped_metrics"],
        "base_rate": 0.40,
        "remediation": "Set sampling to 100% for error spans; use tail-based sampling",
    },
    {
        "id": "C8.10", "category_id": "C8", "category": "Operational",
        "label": "Single-Region Deployment → Region Outage Kills All Agents",
        "signals": ["very_high_latency", "error_response"],
        "base_rate": 0.25,
        "remediation": "Deploy active-active multi-region; use latency-based DNS routing",
    },
    {
        "id": "C8.11", "category_id": "C8", "category": "Operational",
        "label": "No Circuit Breaker for Session DB → Timeout Cascade",
        "signals": ["load_fail", "high_error_rate"],
        "base_rate": 0.35, "requires_session_db": True,
        "remediation": "Add circuit breaker around session DB with half-open retry and fallback",
    },
    {
        "id": "C8.12", "category_id": "C8", "category": "Operational",
        "label": "Logging PII → Compliance Violation",
        "signals": ["pii_detected", "security_fail"],
        "base_rate": 0.45,
        "remediation": "Scrub PII before logging; use field-level masking in structured logs",
    },
    {
        "id": "C8.13", "category_id": "C8", "category": "Operational",
        "label": "No Max Tool Call Depth → Infinite Loop",
        "signals": ["very_high_latency", "consistency_fail"],
        "base_rate": 0.35, "requires_multi_agent": True,
        "remediation": "Set MAX_TOOL_DEPTH constant; raise ToolDepthExceeded after limit",
    },
    {
        "id": "C8.14", "category_id": "C8", "category": "Operational",
        "label": "Missing Alert on Cost Anomaly",
        "signals": ["rate_limit_error"],
        "base_rate": 0.40,
        "remediation": "Set cost budget alerts in cloud console; add token-spend metric to dashboard",
    },
]


# ── Signal → detailed narrative (used in reason generation) ───────────────────

_SIGNAL_NARRATIVE: Dict[str, str] = {
    "hallucination_fail":     "the model generated factually incorrect or unsupported content",
    "answer_relevancy_fail":  "responses failed to adequately address the user's actual question",
    "usefulness_fail":        "outputs were rated as unhelpful or insufficiently actionable",
    "completeness_fail":      "responses omitted required information or gave partial answers",
    "consistency_fail":       "contradictions appeared across turns or between the response and source context",
    "judge_fail":             "the LLM judge determined that responses did not meet quality criteria",
    "pii_detected":           "personally identifiable information was found in model outputs",
    "injection_detected":     "prompt injection attempts successfully altered model behaviour",
    "toxicity_fail":          "generated content was flagged for harmful or offensive language",
    "bias_fail":              "outputs displayed discriminatory or stereotyping patterns",
    "attack_resistance_fail": "the model failed to resist adversarial inputs or jailbreak attempts",
    "security_fail":          "one or more security evaluation checks failed",
    "rag_faithfulness_fail":  "generated answers were not grounded in the retrieved source documents",
    "rag_recall_fail":        "relevant information was missing from retrieved chunks or not used",
    "error_response":         "the endpoint returned error responses instead of valid answers",
    "rate_limit_error":       "HTTP 429 rate-limit errors were encountered during testing",
    "timeout_error":          "requests timed out before the model returned a response",
    "high_latency":           "average response latency exceeded 3 seconds",
    "very_high_latency":      "extreme latency was observed — average above 8 s or P95 above 10 s",
    "load_fail":              "the system produced errors or degraded under concurrent load",
    "high_error_rate":        "more than 10% of requests resulted in errors",
    "skipped_metrics":        "several evaluation metrics were skipped due to tool or API failures",
    "multi_turn_fail":        "multi-turn conversations ended in abandoned or frustrated states",
    "long_prompt_fail":       "long user prompts triggered errors, suggesting context window pressure",
}

# ── Category → root cause mechanism explanation ────────────────────────────────

_CATEGORY_CONTEXT: Dict[str, str] = {
    "AI / Model":
        "This class of failure originates in the model's generative behaviour — including its "
        "prompting configuration, temperature, context window management, and instruction-following "
        "capability. These issues are typically resolved through prompt engineering, few-shot examples, "
        "or output validation.",
    "RAG-Specific":
        "This failure is specific to the retrieval-augmented generation pipeline — either the retrieval "
        "step is returning irrelevant or incomplete content, or the model is not faithfully grounding "
        "its answers in what was retrieved. Both the chunking strategy and the similarity threshold "
        "are common culprits.",
    "Integration & API":
        "This failure occurs at the boundary between your application and external APIs — the LLM "
        "provider, downstream tool APIs, or the API gateway — typically manifesting as errors, "
        "timeouts, or rate-limit exhaustion. Resilience patterns (retries, circuit breakers, "
        "backoff) are the primary remediation path.",
    "Security":
        "This is a security-class failure where the AI system is either leaking sensitive data, "
        "being manipulated by adversarial inputs, or operating with insufficient access controls. "
        "These failures can have regulatory and trust implications beyond pure quality degradation.",
    "Latency & Performance":
        "This is a performance failure where excessive latency or resource contention is degrading "
        "user experience or causing downstream timeouts. Root causes range from cold starts and "
        "sequential tool calls to large context windows and unoptimised retrieval.",
    "TPM / Rate Limiting":
        "This failure stems from token-per-minute (TPM) quota exhaustion — either at the provider "
        "level or due to insufficient per-user rate management in your application. Embedding tokens "
        "and LLM completion tokens both count toward limits.",
    "Database & Session":
        "This failure originates in the session or database layer — inconsistent state, connection "
        "pool saturation, TTL expiry, or read-replica lag are causing conversation-level failures "
        "that appear as quality or consistency problems.",
    "Operational":
        "This is an operational failure — gaps in observability, deployment configuration, or "
        "incident response posture are amplifying the impact of other failure modes. Without "
        "tracing and alerting, these failures are also the hardest to diagnose in production.",
}

# ── Reason builder ─────────────────────────────────────────────────────────────

_ARCH_FLAG_NOTES: Dict[str, str] = {
    "no_retry":          "the absence of retry logic means transient failures are never recovered from automatically",
    "no_circuit_breaker": "no circuit breaker is configured, so a failing dependency will cascade errors to all downstream callers",
    "no_rate_limiting":  "per-user rate limiting is absent, increasing the risk that a single session exhausts the shared quota",
    "synchronous_calls": "synchronous API calls block the entire request thread while waiting on slow dependencies",
    "single_region":     "single-region deployment creates a single point of failure — any regional outage affects all users",
    "no_dlq":            "without a dead-letter queue, failed tasks are silently discarded with no retry or alerting",
    "no_streaming":      "without streaming, the user receives no output until the full response is generated, amplifying latency perception",
    "no_cache":          "without a response cache, every request hits the LLM even when an identical answer exists",
    "no_monitoring":     "without monitoring or tracing, failure patterns are invisible until users report them",
}

_ARCH_BOOST_MAP: Dict[str, List[str]] = {
    "no_retry":           ["C3.9", "C3.4", "C6.3"],
    "no_circuit_breaker": ["C3.7", "C3.12", "C8.11"],
    "no_rate_limiting":   ["C3.1", "C4.14", "C6.2"],
    "no_cache":           ["C3.17"],
    "no_streaming":       ["C5.12"],
    "synchronous_calls":  ["C3.18", "C5.2"],
    "single_region":      ["C8.10", "C8.6"],
}


def _build_reason(
    point: Dict[str, Any],
    signals: Dict[str, int],
    total_failed: int,
    prob: float,
    affected: int,
    extra_flags: Set[str],
) -> str:
    """Generate a detailed narrative explanation for a root cause finding."""
    point_signals = point.get("signals", [])
    matched = [(s, signals[s]) for s in point_signals if signals.get(s, 0) > 0]

    prob_pct = int(prob * 100)
    category = point["category"]

    if prob_pct >= 70:
        severity_phrase = "a high-confidence root cause"
    elif prob_pct >= 45:
        severity_phrase = "a moderately likely contributing factor"
    else:
        severity_phrase = "a possible contributing factor"

    cat_context = _CATEGORY_CONTEXT.get(category, "")
    parts: List[str] = []

    # Opening sentence with category mechanism
    parts.append(
        f'This failure point is {severity_phrase} at {prob_pct}% probability. '
        f'{cat_context}'
    )

    # Signal-based evidence narrative
    if matched:
        signal_parts = []
        for sig, count in matched:
            narrative = _SIGNAL_NARRATIVE.get(sig, sig.replace("_", " "))
            if count > 1:
                signal_parts.append(f"{narrative} ({count} times)")
            else:
                signal_parts.append(narrative)

        if len(signal_parts) == 1:
            parts.append(f"The observed evidence is that {signal_parts[0]}.")
        elif len(signal_parts) == 2:
            parts.append(
                f"The observed evidence includes: {signal_parts[0]}, and {signal_parts[1]}."
            )
        else:
            joined = "; ".join(signal_parts[:-1]) + f"; and {signal_parts[-1]}"
            parts.append(f"The observed evidence includes: {joined}.")
    else:
        parts.append(
            "No direct failure signals were matched in this run, but the architecture profile "
            "and base-rate prior for this failure class elevate its probability."
        )

    # Affected proportion
    if affected > 0 and total_failed > 0:
        pct = round(100.0 * affected / max(total_failed, 1))
        parts.append(
            f"This root cause may be responsible for approximately {pct}% of the "
            f"{total_failed} failing test case{'s' if total_failed != 1 else ''}."
        )
    elif total_failed > 0:
        parts.append(
            f"The failure is inferred from the architecture profile rather than "
            f"direct signal matches across the {total_failed} failing test cases."
        )

    # Architecture-specific amplifiers relevant to this point
    relevant_arch_notes: List[str] = []
    for flag, ids in _ARCH_BOOST_MAP.items():
        if flag in extra_flags and point["id"] in ids:
            note = _ARCH_FLAG_NOTES.get(flag)
            if note:
                relevant_arch_notes.append(note)

    if relevant_arch_notes:
        if len(relevant_arch_notes) == 1:
            parts.append(f"Additionally, {relevant_arch_notes[0]}.")
        else:
            joined = "; ".join(relevant_arch_notes[:-1]) + f"; and {relevant_arch_notes[-1]}"
            parts.append(f"Additionally, {joined}.")

    return " ".join(parts)


# ── Keyword → extra architecture flag map (for additional_architecture_notes) ─

_KEYWORD_FLAGS: Dict[str, str] = {
    # Message queues
    "kafka":              "has_message_queue",
    "rabbitmq":           "has_message_queue",
    "sqs":                "has_message_queue",
    "pubsub":             "has_message_queue",
    "kinesis":            "has_message_queue",
    "celery":             "has_message_queue",
    # Serverless
    "lambda":             "serverless",
    "azure function":     "serverless",
    "cloud function":     "serverless",
    "cloud run":          "serverless",
    "fargate":            "serverless",
    "serverless":         "serverless",
    # Missing resilience
    "no retry":           "no_retry",
    "without retry":      "no_retry",
    "no circuit breaker": "no_circuit_breaker",
    "no auth":            "no_auth",
    "no authentication":  "no_auth",
    "no rate limit":      "no_rate_limiting",
    "no monitoring":      "no_monitoring",
    "no logging":         "no_logging",
    "no cache":           "no_cache",
    "no streaming":       "no_streaming",
    "no dlq":             "no_dlq",
    "no dead letter":     "no_dlq",
    # Infrastructure patterns
    "synchronous":        "synchronous_calls",
    "single region":      "single_region",
    "single-region":      "single_region",
    "monolith":           "monolith",
    "microservice":       "microservices",
    "multi region":       "multi_region",
    "multi-region":       "multi_region",
    # Vector DBs
    "pinecone":           "has_vector_db",
    "weaviate":           "has_vector_db",
    "qdrant":             "has_vector_db",
    "faiss":              "has_vector_db",
    "chroma":             "has_vector_db",
    "opensearch":         "has_vector_db",
    # Session / relational DBs
    "redis":              "has_redis",
    "postgresql":         "has_rdbms",
    "mysql":              "has_rdbms",
    "mongodb":            "has_nosql",
    "dynamodb":           "has_nosql",
    "cosmosdb":           "has_nosql",
    # Caching
    "memcached":          "has_cache",
    "cdn":                "has_cache",
    "cloudfront":         "has_cache",
    "varnish":            "has_cache",
    # API gateways
    "api gateway":        "has_api_gateway",
    "kong":               "has_api_gateway",
    "nginx":              "has_api_gateway",
    "apim":               "has_api_gateway",
    "traefik":            "has_api_gateway",
    # Monitoring
    "datadog":            "has_monitoring",
    "cloudwatch":         "has_monitoring",
    "prometheus":         "has_monitoring",
    "grafana":            "has_monitoring",
    "newrelic":           "has_monitoring",
    "jaeger":             "has_monitoring",
    "opentelemetry":      "has_monitoring",
    # Auth
    "oauth":              "has_auth",
    "jwt":                "has_auth",
    "saml":               "has_auth",
    "auth0":              "has_auth",
    "okta":               "has_auth",
}


def _parse_arch_notes(notes: str) -> Set[str]:
    """Return a set of extra architecture flag strings from free-form notes."""
    flags: Set[str] = set()
    lower = notes.lower()
    for keyword, flag in _KEYWORD_FLAGS.items():
        if keyword in lower:
            flags.add(flag)
    return flags


# ── Architecture relevance filter ─────────────────────────────────────────

def _is_relevant(point: Dict[str, Any], config: RunConfig, extra_flags: Set[str]) -> bool:
    """Return True if this failure point is relevant given the architecture."""
    if point.get("requires_rag") and not config.is_rag:
        return False
    if point.get("requires_serverless"):
        is_serverless = (
            config.deployment_type == "serverless"
            or "serverless" in extra_flags
        )
        if not is_serverless:
            return False
    if point.get("requires_session_db"):
        has_db = (
            bool(config.session_db)
            or "has_redis" in extra_flags
            or "has_rdbms" in extra_flags
            or "has_nosql" in extra_flags
        )
        if not has_db:
            return False
    if point.get("requires_vector_db"):
        has_vdb = bool(config.vector_db) or "has_vector_db" in extra_flags
        if not has_vdb:
            return False
    if point.get("requires_multi_agent"):
        if config.application_type.value != "multi_agent":
            return False
    if point.get("requires_message_queue"):
        has_mq = (
            bool(config.message_queue)
            or "has_message_queue" in extra_flags
        )
        if not has_mq:
            return False
    return True


# ── Signal extractor ───────────────────────────────────────────────────────

def _extract_signals(
    metric_results: List[MetricResult],
    conversations: List[Conversation],
    perf_metrics: Dict[str, Any],
    load_metrics: Dict[str, Any],
) -> Tuple[Dict[str, int], int]:
    """
    Count occurrences of each signal from evaluation results.
    Returns (signal_counts, total_failed_prompts).
    """
    sig: Dict[str, int] = {k: 0 for k in _S}
    failed_results = [r for r in metric_results if not r.passed and not r.skipped]
    total_failed = len(failed_results)

    for r in failed_results:
        mn = r.metric_name.lower()

        if "hallucination" in mn:
            sig["hallucination_fail"] += 1
        if "relevancy" in mn or "relevance" in mn:
            sig["answer_relevancy_fail"] += 1
        if "usefulness" in mn:
            sig["usefulness_fail"] += 1
        if "completeness" in mn:
            sig["completeness_fail"] += 1
        if "consistency" in mn:
            sig["consistency_fail"] += 1
        if "judge" in mn or "correctness" in mn or "accuracy" in mn:
            sig["judge_fail"] += 1
        if "pii" in mn and r.pii_detected:
            sig["pii_detected"] += 1
        if "injection" in mn or "prompt_injection" in mn:
            sig["injection_detected"] += 1
        if "toxicity" in mn or "toxic" in mn:
            sig["toxicity_fail"] += 1
        if "bias" in mn:
            sig["bias_fail"] += 1
        if "attack" in mn or "resistance" in mn:
            sig["attack_resistance_fail"] += 1
        if r.superset == "security":
            sig["security_fail"] += 1
        if "faithfulness" in mn:
            sig["rag_faithfulness_fail"] += 1
        if "recall" in mn or "precision" in mn or "context_recall" in mn:
            sig["rag_recall_fail"] += 1

    # Skipped metrics count
    skipped_count = sum(1 for r in metric_results if r.skipped)
    if skipped_count > 3:
        sig["skipped_metrics"] = skipped_count

    # Conversation-level signals
    error_count = 0
    latency_values: List[float] = []
    rate_limit_count = 0
    timeout_count = 0

    for conv in conversations:
        for turn in conv.turns:
            if turn.is_error_response:
                error_count += 1
                resp_lower = turn.response.lower()
                if "429" in turn.response or "rate limit" in resp_lower or "too many" in resp_lower:
                    rate_limit_count += 1
                if "504" in turn.response or "timeout" in resp_lower or "timed out" in resp_lower:
                    timeout_count += 1
            latency_values.append(turn.latency_ms)

        # Multi-turn failure: conversation has multiple turns and ended badly
        if len(conv.turns) > 1 and conv.final_state and conv.final_state.value in ("abandoned", "frustrated"):
            sig["multi_turn_fail"] += 1

    if error_count > 0:
        sig["error_response"] = error_count
    if rate_limit_count > 0:
        sig["rate_limit_error"] = rate_limit_count
    if timeout_count > 0:
        sig["timeout_error"] = timeout_count

    # Latency signals (from conversation turns)
    if latency_values:
        avg_lat = sum(latency_values) / len(latency_values)
        if avg_lat > 3000:
            sig["high_latency"] = int(avg_lat)
        if avg_lat > 8000:
            sig["very_high_latency"] = int(avg_lat)

    # Latency signals (from perf metrics, overrides if stronger evidence)
    p95 = perf_metrics.get("p95_latency_ms", 0.0)
    if p95 > 5000:
        sig["high_latency"] = max(sig["high_latency"], int(p95))
    if p95 > 10000:
        sig["very_high_latency"] = max(sig["very_high_latency"], int(p95))

    perf_error_rate = perf_metrics.get("error_rate", 0.0)
    if perf_error_rate > 0.10:
        sig["high_error_rate"] = int(perf_error_rate * 100)

    # Load test signals
    if not load_metrics.get("passed", True) or load_metrics.get("error_rate", 0.0) > 0.15:
        sig["load_fail"] += load_metrics.get("errors", 1)

    # Long prompt detection: prompts longer than 1500 chars
    for conv in conversations:
        for turn in conv.turns:
            if len(turn.query) > 1500 and turn.is_error_response:
                sig["long_prompt_fail"] += 1
                break

    return sig, total_failed


# ── Point scorer ──────────────────────────────────────────────────────────

def _score_point(
    point: Dict[str, Any],
    signals: Dict[str, int],
    total_failed: int,
    config: RunConfig,
    extra_flags: Set[str],
) -> Tuple[float, int, List[str]]:
    """
    Score a failure point. Returns (probability, affected_count, evidence_list).

    probability = base_rate * 0.25  +  signal_contribution * 0.75
    signal_contribution = (weighted matched signals) / total_failed   (capped at 1.0)
    """
    point_signals = point.get("signals", [])
    matched_count = 0
    evidence: List[str] = []

    for sig_key in point_signals:
        count = signals.get(sig_key, 0)
        if count > 0:
            matched_count += count
            # Build human-readable evidence string
            label_map = {
                "hallucination_fail":    f"hallucination metric failed ({count}× )",
                "answer_relevancy_fail": f"answer relevancy failed ({count}× )",
                "usefulness_fail":       f"usefulness metric failed ({count}× )",
                "completeness_fail":     f"completeness metric failed ({count}× )",
                "consistency_fail":      f"consistency metric failed ({count}× )",
                "judge_fail":            f"LLM judge failed ({count}× )",
                "pii_detected":          f"PII detected in {count} response(s)",
                "injection_detected":    f"prompt injection flagged ({count}× )",
                "toxicity_fail":         f"toxicity metric failed ({count}× )",
                "bias_fail":             f"bias metric failed ({count}× )",
                "attack_resistance_fail": f"attack resistance failed ({count}× )",
                "security_fail":         f"security evaluation failed ({count}× )",
                "rag_faithfulness_fail": f"RAG faithfulness failed ({count}× )",
                "rag_recall_fail":       f"RAG recall/precision failed ({count}× )",
                "error_response":        f"{count} error response(s) from endpoint",
                "rate_limit_error":      f"rate-limit error detected ({count}× )",
                "timeout_error":         f"timeout error detected ({count}× )",
                "high_latency":          f"high avg latency ({count} ms)",
                "very_high_latency":     f"very high avg latency ({count} ms)",
                "load_fail":             f"load test failure ({count} errors)",
                "high_error_rate":       f"error rate {count}%",
                "skipped_metrics":       f"{count} metric(s) skipped (tool/API failure)",
                "multi_turn_fail":       f"multi-turn conversation failures ({count}× )",
                "long_prompt_fail":      f"long-prompt errors ({count}× )",
            }
            evidence.append(label_map.get(sig_key, sig_key))

    # Architecture-based bonus signals (from extra_flags)
    arch_boosts = {
        "no_retry":        ["C3.9", "C3.4", "C6.3"],
        "no_circuit_breaker": ["C3.7", "C3.12", "C8.11"],
        "no_rate_limiting": ["C3.1", "C4.14", "C6.2"],
        "no_monitoring":   ["C8.1", "C8.9"],
        "no_cache":        ["C3.17"],
        "no_streaming":    ["C5.12"],
        "synchronous_calls": ["C3.18", "C5.2"],
        "single_region":   ["C8.10", "C8.6"],
    }
    boost = 0.0
    for flag, point_ids in arch_boosts.items():
        if flag in extra_flags and point["id"] in point_ids:
            boost += 0.10
            evidence.append(f"architecture note: {flag.replace('_', ' ')}")

    if total_failed == 0:
        signal_contribution = 0.0
    else:
        signal_contribution = min(1.0, matched_count / max(total_failed, 1))

    base_rate = point.get("base_rate", 0.30)
    probability = min(0.97, base_rate * 0.25 + signal_contribution * 0.75 + boost)

    # If zero signals matched, cap probability low (base-rate prior only)
    if matched_count == 0:
        probability = min(0.15, base_rate * 0.20)

    affected_count = min(matched_count, total_failed)
    return round(probability, 3), affected_count, evidence


# ── Main entry point ───────────────────────────────────────────────────────

def run_rca(
    metric_results: List[MetricResult],
    conversations: List[Conversation],
    config: RunConfig,
    perf_metrics: Dict[str, Any],
    load_metrics: Dict[str, Any],
    top_n: int = 10,
) -> RCAReport:
    """
    Run full RCA. Returns RCAReport with top-N probable root causes.

    Steps:
      1. Extract signals from eval results + conversations + metrics
      2. Parse additional_architecture_notes into extra flags
      3. Filter taxonomy to architecture-relevant points
      4. Score each relevant point
      5. Sort by probability; return top_n
    """
    signals, total_failed = _extract_signals(
        metric_results, conversations, perf_metrics, load_metrics
    )

    extra_flags = _parse_arch_notes(config.additional_architecture_notes)

    # Also derive flags from structured RunConfig fields
    if not config.has_retry_logic:
        extra_flags.add("no_retry")
    if not config.has_circuit_breaker:
        extra_flags.add("no_circuit_breaker")
    if not config.has_rate_limiting:
        extra_flags.add("no_rate_limiting")
    if config.deployment_type == "serverless":
        extra_flags.add("serverless")
    if config.session_db:
        extra_flags.add("has_session_db")
    if config.vector_db:
        extra_flags.add("has_vector_db")

    # Filter taxonomy
    relevant = [p for p in TAXONOMY if _is_relevant(p, config, extra_flags)]
    filtered_count = len(TAXONOMY) - len(relevant)

    # Score each relevant point
    scored: List[Tuple[float, int, List[str], Dict]] = []
    for point in relevant:
        prob, affected, evidence = _score_point(point, signals, total_failed, config, extra_flags)
        if prob > 0.0:
            scored.append((prob, affected, evidence, point))

    # Sort by probability descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build top-N findings
    findings: List[RCAFinding] = []
    for rank, (prob, affected, evidence, point) in enumerate(scored[:top_n], start=1):
        reason = _build_reason(point, signals, total_failed, prob, affected, extra_flags)
        findings.append(RCAFinding(
            rank=rank,
            id=point["id"],
            label=point["label"],
            category=point["category"],
            category_id=point["category_id"],
            probability=prob,
            affected_count=affected,
            evidence=evidence if evidence else ["no direct signal — inferred from architecture profile"],
            remediation=point.get("remediation", ""),
            reason=reason,
        ))

    # Architecture summary snapshot
    arch_summary = {
        "application_type":    config.application_type.value,
        "is_rag":              config.is_rag,
        "deployment_type":     config.deployment_type,
        "vector_db":           config.vector_db or "none",
        "session_db":          config.session_db or "none",
        "has_rate_limiting":   config.has_rate_limiting,
        "has_retry_logic":     config.has_retry_logic,
        "has_circuit_breaker": config.has_circuit_breaker,
        "extra_flags_detected": sorted(extra_flags),
    }

    # Clean signal summary (remove zero-count signals)
    signal_summary = {k: v for k, v in signals.items() if v > 0}

    return RCAReport(
        total_failed=total_failed,
        total_analyzed=len(metric_results),
        relevant_points=len(relevant),
        filtered_points=filtered_count,
        architecture_summary=arch_summary,
        signal_summary=signal_summary,
        top_causes=findings,
    )
