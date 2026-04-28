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
{rag_section}
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
{rag_rules}
- Message 1: opening — how they'd first approach the app to reach their goal
- Message 2: alternative angle — a different way to ask for the same goal
- Message 3: follow-up — what they'd ask after getting a vague/partial response

Return JSON:
{{
  "prompts": [
    {{
      "text": "<message in persona's voice>",
      "expected_behavior": "<what a CORRECT, COMPLETE response looks like — be specific, cite facts from the knowledge base if RAG mode>",
      "turn": 1
    }},
    {{
      "text": "<alternative opening message>",
      "expected_behavior": "<expected correct response>",
      "turn": 1
    }},
    {{
      "text": "<follow-up after vague response>",
      "expected_behavior": "<expected correct response>",
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
    rag_text: str = "",
) -> List[GeneratedPrompt]:
    """
    Generate 3 domain-specific functional test prompts for a persona.
    For RAG mode, rag_text is included so expected_behavior is grounded in the knowledge base.
    """
    # Build RAG-specific sections if knowledge base text is provided
    rag_section = ""
    rag_rules = ""
    if rag_text and rag_text.strip():
        # Increased from 800 to 3000 chars so test cases are grounded in a
        # representative portion of the knowledge base, not just the opening lines.
        snippet = rag_text.strip()[:3000]
        rag_section = f"- Knowledge Base (excerpt):\n{snippet}\n"
        rag_rules = "- The expected_behavior MUST be based on the actual content in the knowledge base above\n- Do NOT hallucinate facts — only reference what is in the knowledge base\n"

    prompt = FUNCTIONAL_PROMPT.format(
        application_type=profile.application_type.value,
        domain=profile.domain,
        use_cases=", ".join(profile.use_cases[:4]),
        boundaries=", ".join(profile.boundaries[:3]) or "none specified",
        rag_section=rag_section,
        rag_rules=rag_rules,
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

    # ── Priority 1: use multi_turn_scenario from rich persona generation ──
    # These are pre-crafted literal prompts with real artifacts (typos, paste
    # fragments, mixed languages) — far more diverse than LLM-generated generics.
    if persona.multi_turn_scenario:
        for turn in persona.multi_turn_scenario[:3]:
            text = turn.get("prompt", "").strip()
            if text:
                generated.append(GeneratedPrompt(
                    persona_id=persona.persona_id,
                    test_class=TestClass.FUNCTIONAL,
                    text=text,
                    expected_behavior="",
                    turn_number=turn.get("turn", 1),
                ))
        # If scenario gave us prompts, also add example_prompts as additional tests
        for ep in persona.entry_points[:2]:
            if ep and ep not in {g.text for g in generated}:
                generated.append(GeneratedPrompt(
                    persona_id=persona.persona_id,
                    test_class=TestClass.FUNCTIONAL,
                    text=ep,
                    expected_behavior="",
                    turn_number=1,
                ))
        if generated:
            return generated

    # ── Priority 2: LLM-generated prompts (grounded in app profile / RAG KB) ──
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
    except Exception as e:
        print(f"[FunctionalGen] LLM prompt generation failed for persona '{persona.name}' "
              f"(domain: {profile.domain}) — falling back to entry_points. Error: {e}")

    # ── Priority 3: entry_points from persona ─────────────────────────────
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
    except Exception as e:
        print(f"[FunctionalGen] Performance prompt generation failed for persona '{persona.name}' — using entry_point. Error: {e}")
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
    rag_text: str = "",
    max_prompts: int = 0,   # Fix 35: 0 = no cap; >0 = cap total prompts to this value
) -> List[GeneratedPrompt]:
    """
    Generate functional prompts for all personas in parallel via LLM.
    In RAG mode: rag_text grounds the generated questions in the knowledge base.
    Ground truth Q&A pairs are handled separately in Stream 2 (pipeline.py).

    Fix 35: max_prompts caps the total output when config.num_scenarios is set.
    """
    sem = asyncio.Semaphore(5)

    async def _bounded(persona):
        async with sem:
            return await generate_functional_prompts(persona, profile, llm_client, rag_text=rag_text)

    batches = await asyncio.gather(*[_bounded(p) for p in personas])
    result = [prompt for batch in batches for prompt in batch]

    if max_prompts > 0:
        result = result[:max_prompts]
    return result
