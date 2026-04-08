"""
Stage 1a: Fishbone Coverage Builder.
Generates a systematic 5-dimension persona slot matrix so every run covers
novice/expert, calm/frustrated, genuine/adversarial users — not random picks.

Sourced from new metron-backend/app/stage2_personas/fishbone_builder.py.
"""

from __future__ import annotations
import hashlib
from typing import Dict, List

from core.config import HIGH_SECURITY_DOMAINS, HIGH_TRAFFIC_DOMAINS
from core.models import AppProfile, ExpertiseLevel, EmotionalState, PersonaIntent

# Fixed dimension values
EXPERTISE_LEVELS  = [e.value for e in ExpertiseLevel]
EMOTIONAL_STATES  = [e.value for e in EmotionalState]
INTENT_TYPES      = [i.value for i in PersonaIntent]


def build_slots(profile: AppProfile, num_personas: int = 6) -> List[Dict[str, str]]:
    """
    Return a list of persona slot dicts, each containing:
      user_type, expertise, emotional_state, intent, goal_type
    Slots are domain-weighted (security domains → more adversarial, etc.)
    """
    domain    = profile.domain.lower()
    user_types = profile.user_types or ["general user"]
    use_cases  = profile.use_cases  or ["general assistance"]

    # Domain-specific slot counts
    if domain in HIGH_SECURITY_DOMAINS:
        adversarial_count = min(len(user_types), 4)
        edge_case_count   = 3
        emotional_count   = 2
    elif domain in HIGH_TRAFFIC_DOMAINS:
        adversarial_count = min(2, len(user_types))
        edge_case_count   = 2
        emotional_count   = 4
    else:
        adversarial_count = min(len(user_types), 3)
        edge_case_count   = 2
        emotional_count   = 3

    slots: List[Dict[str, str]] = []

    # Group A: genuine users, cross expertise × user_type (calm)
    for i, ut in enumerate(user_types[:3]):
        expertise = EXPERTISE_LEVELS[i % len(EXPERTISE_LEVELS)]
        slots.append({
            "user_type":     ut,
            "expertise":     expertise,
            "emotional_state": "calm",
            "intent":        "genuine",
            "goal_type":     use_cases[0] if use_cases else "general assistance",
        })

    # Group B: emotional / frustrated genuine users
    for i in range(min(emotional_count, len(user_types))):
        ut = user_types[i % len(user_types)]
        slots.append({
            "user_type":     ut,
            "expertise":     "novice",
            "emotional_state": "frustrated",
            "intent":        "genuine",
            "goal_type":     use_cases[i % len(use_cases)],
        })
    # One urgent intermediate
    slots.append({
        "user_type":     user_types[0],
        "expertise":     "intermediate",
        "emotional_state": "urgent",
        "intent":        "genuine",
        "goal_type":     use_cases[-1] if use_cases else "general assistance",
    })

    # Group C: adversarial personas (cycled expertise)
    for i in range(adversarial_count):
        ut = user_types[i % len(user_types)]
        expertise = EXPERTISE_LEVELS[i % len(EXPERTISE_LEVELS)]
        slots.append({
            "user_type":     ut,
            "expertise":     expertise,
            "emotional_state": "calm",
            "intent":        "adversarial",
            "goal_type":     "security testing",
        })

    # Group D: edge cases
    edge_types = [
        ("novice", "wrong assumption about capabilities"),
        ("expert", "edge case boundary testing"),
        ("intermediate", "ambiguous or multi-part request"),
    ]
    for i in range(min(edge_case_count, len(edge_types))):
        exp, goal = edge_types[i]
        slots.append({
            "user_type":     user_types[i % len(user_types)],
            "expertise":     exp,
            "emotional_state": "calm",
            "intent":        "edge_case",
            "goal_type":     goal,
        })

    # Group E: goal-type diversity (one per use case, genuine calm)
    for i, uc in enumerate(use_cases[1:3]):   # up to 2 extra
        slots.append({
            "user_type":     user_types[i % len(user_types)],
            "expertise":     "intermediate",
            "emotional_state": "calm",
            "intent":        "genuine",
            "goal_type":     uc,
        })

    # Trim or pad to num_personas
    if len(slots) > num_personas:
        # Keep balance: prioritise group A (genuine), then adversarial, then others
        genuine    = [s for s in slots if s["intent"] == "genuine"]
        adversarial = [s for s in slots if s["intent"] == "adversarial"]
        edge       = [s for s in slots if s["intent"] == "edge_case"]
        combined   = genuine + adversarial + edge
        slots      = combined[:num_personas]

    return slots


def slot_id(domain: str, slot: Dict[str, str]) -> str:
    """Deterministic SHA-256 ID for a slot so the same slot maps to the same persona."""
    key = f"{domain}|{slot['user_type']}|{slot['expertise']}|{slot['emotional_state']}|{slot['intent']}|{slot['goal_type']}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
