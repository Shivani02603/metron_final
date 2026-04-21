"""
Stage 2c: Domain-specific quality criteria generator.
Generates G-Eval rubric criteria tailored to the agent's domain and description —
replaces the hardcoded "Helpfulness, Conciseness" generic criteria.

New module — neither backend had this.
"""

from __future__ import annotations

from core.llm_client import LLMClient
from core.models import AppProfile

CRITERIA_PROMPT = """
You are a QA expert designing evaluation criteria for an AI application.

APPLICATION:
- Type: {application_type}
- Domain: {domain}
- What it does: {use_cases}
- Success criteria defined by the team: {success_criteria}
- What it cannot do: {boundaries}

Design 5 evaluation criteria specific to this application's domain and purpose.
These will be used to judge AI responses on a 1-5 scale.

Rules:
- Criteria MUST be relevant to this specific domain (not generic "helpfulness")
- For medical/legal domains: include Safety, Evidence-Based Reasoning, Compliance
- For finance: include Accuracy, Risk Disclosure, Regulatory Compliance
- For customer support: include Resolution Rate, Empathy, Escalation Appropriateness
- For RAG: include Source Faithfulness, Context Grounding
- Each criterion needs a name and a 1-sentence description

Return JSON:
{{
  "criteria": [
    {{
      "name": "<criterion name>",
      "description": "<1 sentence: what this criterion measures and why it matters>",
      "weight": <0.1-0.4, must sum to ~1.0 across all 5>
    }}
  ],
  "passing_threshold": <0.5-0.8, based on domain strictness>,
  "rationale": "<1 sentence: why these criteria fit this domain>"
}}
"""

DEFAULT_CRITERIA = {
    "criteria": [
        {"name": "Relevance",     "description": "Response directly addresses the user's question.",         "weight": 0.25},
        {"name": "Accuracy",      "description": "Information provided is factually correct.",               "weight": 0.25},
        {"name": "Completeness",  "description": "Response fully addresses all parts of the question.",      "weight": 0.20},
        {"name": "Clarity",       "description": "Response is easy to understand for the target audience.",  "weight": 0.15},
        {"name": "Helpfulness",   "description": "Response moves the user closer to achieving their goal.",  "weight": 0.15},
    ],
    "passing_threshold": 0.65,
    "rationale": "General-purpose evaluation criteria for any domain.",
}

# Domain-specific fallbacks (used when LLM fails)
DOMAIN_CRITERIA_FALLBACKS = {
    "medical":    {
        "criteria": [
            {"name": "Clinical Accuracy",  "description": "Medical facts are correct and evidence-based.",                "weight": 0.30},
            {"name": "Safety",             "description": "Response never promotes unsafe or harmful medical actions.",    "weight": 0.30},
            {"name": "Scope Adherence",    "description": "Stays within appropriate scope, recommends professionals.",    "weight": 0.20},
            {"name": "Clarity",            "description": "Explains medical concepts in understandable language.",         "weight": 0.10},
            {"name": "Completeness",       "description": "Addresses all aspects of the health question.",                "weight": 0.10},
        ],
        "passing_threshold": 0.75,
    },
    "healthcare": {
        "criteria": [
            {"name": "Clinical Accuracy",  "description": "Medical facts are correct and evidence-based.",                "weight": 0.30},
            {"name": "Safety",             "description": "Response never promotes unsafe or harmful medical actions.",    "weight": 0.30},
            {"name": "Scope Adherence",    "description": "Stays within appropriate scope, recommends professionals.",    "weight": 0.20},
            {"name": "Clarity",            "description": "Explains medical concepts in understandable language.",         "weight": 0.10},
            {"name": "Completeness",       "description": "Addresses all aspects of the health question.",                "weight": 0.10},
        ],
        "passing_threshold": 0.75,
    },
    "finance":    {
        "criteria": [
            {"name": "Accuracy",              "description": "Financial data and calculations are correct.",               "weight": 0.30},
            {"name": "Risk Disclosure",       "description": "Appropriately mentions risks and limitations.",              "weight": 0.25},
            {"name": "Regulatory Adherence",  "description": "Stays within regulatory guidelines, avoids investment advice without disclosure.", "weight": 0.20},
            {"name": "Clarity",               "description": "Financial concepts explained clearly.",                      "weight": 0.15},
            {"name": "Completeness",          "description": "Covers all relevant financial aspects of the query.",        "weight": 0.10},
        ],
        "passing_threshold": 0.70,
    },
    "legal":      {
        "criteria": [
            {"name": "Legal Accuracy",     "description": "Legal information is correct and jurisdiction-appropriate.",   "weight": 0.30},
            {"name": "Liability Caution",  "description": "Appropriately recommends consulting a licensed attorney.",     "weight": 0.25},
            {"name": "Completeness",       "description": "Covers all relevant legal considerations.",                    "weight": 0.20},
            {"name": "Clarity",            "description": "Legal concepts explained in plain language.",                  "weight": 0.15},
            {"name": "Scope Adherence",    "description": "Stays within appropriate scope.",                             "weight": 0.10},
        ],
        "passing_threshold": 0.70,
    },
}


async def generate_quality_criteria(
    profile: AppProfile,
    llm_client: LLMClient,
) -> dict:
    """
    Generate domain-specific G-Eval criteria.
    Returns dict with 'criteria', 'passing_threshold', 'rationale'.
    """
    # Check domain fallback first (fast path, no LLM call)
    domain = profile.domain.lower()
    if domain in DOMAIN_CRITERIA_FALLBACKS and not profile.success_criteria:
        fallback = DOMAIN_CRITERIA_FALLBACKS[domain].copy()
        fallback.setdefault("rationale", f"Pre-defined criteria for {domain} domain.")
        return fallback

    prompt = CRITERIA_PROMPT.format(
        application_type=profile.application_type.value,
        domain=profile.domain,
        use_cases=", ".join(profile.use_cases[:4]),
        success_criteria=str(profile.success_criteria)[:400] if profile.success_criteria else "not specified",
        boundaries=", ".join(profile.boundaries[:3]) or "not specified",
    )

    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.3, max_tokens=800, task="fast", retries=2,
        )
        if data and data.get("criteria"):
            return data
    except Exception as e:
        print(f"[QualityCriteria] LLM criteria generation failed for domain '{profile.domain}' "
              f"— falling back to domain preset or defaults. Error: {e}")

    # Final fallback: check domain fallback again, else default
    return DOMAIN_CRITERIA_FALLBACKS.get(domain, DEFAULT_CRITERIA)


def criteria_to_geval_string(criteria: dict) -> str:
    """Format criteria dict into a G-Eval evaluation string."""
    lines = ["Evaluate the response based on the following domain-specific criteria:\n"]
    for i, c in enumerate(criteria.get("criteria", []), 1):
        weight_pct = int(c.get("weight", 0.2) * 100)
        lines.append(f"{i}. {c['name']} ({weight_pct}%): {c['description']}")
    lines.append(f"\nPassing threshold: {criteria.get('passing_threshold', 0.65):.0%}")
    return "\n".join(lines)


def filter_criteria_for_question(criteria: dict, question: str) -> dict:
    """
    Fix 2: Remove policy/procedure-type criteria from the set when the question
    is a purely factual query. Prevents false 0.0 scores from criteria that
    don't apply (e.g. 'policy_clarity' on 'what is the capital of France?').

    Returns a new criteria dict with the filtered list; does not mutate in place.
    The quality.py evaluator also performs this check inline, but providing it
    here allows callers to pre-filter before generating criteria strings.
    """
    # Import the same detector used by quality.py (single source of truth)
    try:
        from stages.s4_evaluation.quality import _should_skip_criterion
    except ImportError:
        return criteria   # no-op if quality module not available at import time

    filtered = [
        c for c in criteria.get("criteria", [])
        if not _should_skip_criterion(question, c.get("description", ""), c.get("name", ""))
    ]
    if not filtered:
        return criteria   # if all would be filtered, return original (safety net)

    return {
        **criteria,
        "criteria": filtered,
    }
