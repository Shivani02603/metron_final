"""
Stage 2a: Domain-specific functional test prompt generator.
Generates realistic test prompts in the persona's exact voice and style,
grounded in the agent's domain (not generic "capital of France" questions).

Sourced from new metron-backend/app/stage3_prompts/prompt_generator.py.
"""

from __future__ import annotations
import asyncio
from typing import List

from core.llm_client import LLMClient
from core.models import AppProfile, GeneratedPrompt, Persona, TestClass

FUNCTIONAL_PROMPT = """
Generate 3 realistic test messages for this persona interacting with this AI application.

APPLICATION:
- Type: {application_type}
- Domain: {domain}
- What it does: {use_cases}
- What it CANNOT do: {boundaries}

PERSONA:
- Name: {persona_name}
- Role/Background: {background}
- Expertise: {expertise} in this domain
- Emotional state: {emotional_state}
- Goal: {goal}
- Communication style: {base_style}
- Vocabulary they use: {vocab_prefer}
- Vocabulary they avoid: {vocab_avoid}

Rules:
- Messages MUST be in this persona's exact voice and style
- Messages must be about something relevant to what this app actually does
- Message 1: opening — how they'd first approach the app to reach their goal
- Message 2: alternative angle — a different way to ask for the same goal
- Message 3: follow-up — what they'd ask after getting a vague/partial response

Return JSON:
{{
  "prompts": [
    {{
      "text": "<message in persona's voice>",
      "expected_behavior": "<what a good response looks like for this persona>",
      "turn": 1
    }},
    {{
      "text": "<alternative opening message>",
      "expected_behavior": "<expected good response>",
      "turn": 1
    }},
    {{
      "text": "<follow-up after vague response>",
      "expected_behavior": "<expected good response>",
      "turn": 2
    }}
  ]
}}
"""

PERFORMANCE_PROMPT = """
Generate 3 complex, multi-step test messages that will stress-test the performance
of this AI application.

APPLICATION:
- Type: {application_type}
- Domain: {domain}
- Use cases: {use_cases}

Rules:
- Queries must require multi-step reasoning or retrieval
- Queries must be complex but realistic (a real user in this domain could ask these)
- At least one must have ambiguous phrasing requiring clarification

Return JSON:
{{
  "prompts": [
    {{
      "text": "<complex multi-step query>",
      "expected_behavior": "<expected thorough response>",
      "complexity": "high" | "medium"
    }}
  ]
}}
"""


async def generate_functional_prompts(
    persona: Persona,
    profile: AppProfile,
    llm_client: LLMClient,
) -> List[GeneratedPrompt]:
    """Generate 3 domain-specific functional test prompts for a persona."""
    prompt = FUNCTIONAL_PROMPT.format(
        application_type=profile.application_type.value,
        domain=profile.domain,
        use_cases=", ".join(profile.use_cases[:4]),
        boundaries=", ".join(profile.boundaries[:3]) or "none specified",
        persona_name=persona.name,
        background=persona.background[:200] if persona.background else persona.description,
        expertise=persona.expertise.value,
        emotional_state=persona.emotional_state.value,
        goal=persona.goal,
        base_style=persona.language_model.base_style or "conversational",
        vocab_prefer=", ".join(persona.language_model.vocabulary_prefer[:5]) or "natural language",
        vocab_avoid=", ".join(persona.language_model.vocabulary_avoid[:5]) or "none",
    )

    generated: List[GeneratedPrompt] = []
    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.8, max_tokens=1200, task="fast", retries=2,
        )
        for p in data.get("prompts", [])[:3]:
            if not p.get("text"):
                continue
            generated.append(GeneratedPrompt(
                persona_id=persona.persona_id,
                test_class=TestClass.FUNCTIONAL,
                text=p["text"],
                expected_behavior=p.get("expected_behavior", ""),
                turn_number=p.get("turn", 1),
            ))
    except Exception:
        pass

    # Fallback: use persona's entry_points
    if not generated:
        for i, ep in enumerate(persona.entry_points[:3]):
            generated.append(GeneratedPrompt(
                persona_id=persona.persona_id,
                test_class=TestClass.FUNCTIONAL,
                text=ep,
                expected_behavior="",
                turn_number=1 if i < 2 else 2,
            ))

    return generated


async def generate_performance_prompts(
    persona: Persona,
    profile: AppProfile,
    llm_client: LLMClient,
) -> List[GeneratedPrompt]:
    """Generate complex performance stress-test prompts."""
    prompt = PERFORMANCE_PROMPT.format(
        application_type=profile.application_type.value,
        domain=profile.domain,
        use_cases=", ".join(profile.use_cases[:4]),
    )
    generated: List[GeneratedPrompt] = []
    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.7, max_tokens=800, task="fast", retries=2,
        )
        for p in data.get("prompts", [])[:3]:
            if not p.get("text"):
                continue
            generated.append(GeneratedPrompt(
                persona_id=persona.persona_id,
                test_class=TestClass.PERFORMANCE,
                text=p["text"],
                expected_behavior=p.get("expected_behavior", ""),
            ))
    except Exception:
        if persona.entry_points:
            generated.append(GeneratedPrompt(
                persona_id=persona.persona_id,
                test_class=TestClass.PERFORMANCE,
                text=persona.entry_points[0],
                expected_behavior="",
            ))
    return generated


async def generate_all_functional(
    personas: List[Persona],
    profile: AppProfile,
    llm_client: LLMClient,
) -> List[GeneratedPrompt]:
    """Generate functional prompts for all personas."""
    all_prompts: List[GeneratedPrompt] = []
    for persona in personas:
        prompts = await generate_functional_prompts(persona, profile, llm_client)
        all_prompts.extend(prompts)
        await asyncio.sleep(0.3)
    return all_prompts
