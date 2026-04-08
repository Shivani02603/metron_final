"""
Stage 1c: Coverage Validator.
Checks generated personas against the fishbone dimensions and identifies gaps.
Suggests up to 3 additional slots to fill coverage holes.

Sourced from new metron-backend/app/stage2_personas/coverage_validator.py.
"""

from __future__ import annotations
from typing import Dict, List

from core.llm_client import LLMClient
from core.models import AppProfile, Persona, ExpertiseLevel, EmotionalState, PersonaIntent

VALIDATION_PROMPT = """
You are reviewing test personas for coverage gaps.

APPLICATION: {domain} {application_type}
USER TYPES: {user_types}
USE CASES: {use_cases}

CURRENT PERSONAS:
{persona_summary}

Identify any significant gaps in coverage (missing user types, expertise levels, or use cases).
Return JSON:
{{
  "gaps_found": ["<gap description>"],
  "additional_slots": [
    {{
      "user_type": "<user type>",
      "expertise": "novice" | "intermediate" | "expert",
      "emotional_state": "calm" | "frustrated" | "urgent",
      "intent": "genuine" | "adversarial" | "edge_case",
      "goal_type": "<specific goal>",
      "reason": "<why this slot fills an important gap>"
    }}
  ]
}}

Rules:
- additional_slots must have AT MOST 3 items
- Only suggest truly important gaps, not minor variations
- If coverage is already good, return empty lists
"""


async def validate_and_fill(
    personas: List[Persona],
    profile: AppProfile,
    llm_client: LLMClient,
) -> List[Dict[str, str]]:
    """
    Returns a list of additional fishbone slots to generate.
    Empty list if coverage is already adequate.
    """
    if len(personas) >= 8:
        return []   # skip validation for large sets (already comprehensive)

    persona_summary = "\n".join(
        f"- {p.name}: {p.expertise.value} {p.user_type}, "
        f"{p.emotional_state.value}, {p.intent.value} intent, goal={p.goal[:60]}"
        for p in personas
    )

    prompt = VALIDATION_PROMPT.format(
        domain=profile.domain,
        application_type=profile.application_type.value,
        user_types=", ".join(profile.user_types[:5]),
        use_cases=", ".join(profile.use_cases[:5]),
        persona_summary=persona_summary,
    )

    try:
        data = await llm_client.complete_json(
            prompt,
            temperature=0.3,
            max_tokens=1000,
            retries=2,
        )
        slots = data.get("additional_slots", [])
        # Validate each slot has required keys
        valid = []
        required = {"user_type", "expertise", "emotional_state", "intent", "goal_type"}
        for s in slots[:3]:
            if required.issubset(s.keys()):
                # Validate enum values
                if s["expertise"] in [e.value for e in ExpertiseLevel] and \
                   s["emotional_state"] in [e.value for e in EmotionalState] and \
                   s["intent"] in [i.value for i in PersonaIntent]:
                    valid.append(s)
        return valid
    except Exception:
        return []
