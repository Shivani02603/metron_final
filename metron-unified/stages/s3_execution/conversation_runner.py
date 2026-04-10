"""
Stage 3: Conversation Runner.
Executes real conversations against the target AI endpoint using the persona's
entry_points and a state machine (seeking → frustrated → escalating → satisfied/abandoned).

Combined turn eval+generate: 1 LLM call per turn (evaluates AI response AND generates next message).
Sourced from new metron-backend/app/stage4_execution/conversation_runner.py.
"""

from __future__ import annotations
import asyncio
from datetime import datetime
from typing import List, Optional

from core.adapters.chatbot import AdapterResponse
from core.adapters.chatbot import ChatbotAdapter
from core.adapters.rag import RAGAdapter
from core.adapters.multiagent import MultiAgentAdapter
from core.adapters.form import FormAdapter
from core.llm_client import LLMClient
from core.models import (
    AppProfile, ApplicationType, Conversation, ConversationState,
    ConversationTurn, GeneratedPrompt, Persona, ResponseType, RunConfig, TestClass,
)

MAX_TURNS = 3   # Turn 1 free (entry_point), Turns 2-3 via combined eval+generate

COMBINED_TURN_PROMPT = """
Do two things at once:

1. Evaluate the AI application's response from this persona's perspective.
2. Generate the next message this persona would send (or null if done).

PERSONA:
- Name: {name}
- Goal: {goal}
- Current state: {current_state}
- Patience level: {patience}/5
- Persistence: {persistence}/10
- Escalation trigger: {escalation_trigger}
- Abandon trigger: {abandon_trigger}
- Normal style: {base_style}
- Frustrated style: {frustrated_style}

AI RESPONSE TO EVALUATE: {response}

CONVERSATION SO FAR:
{history}

Return JSON:
{{
  "response_type": "helpful" | "vague" | "deflecting" | "wrong" | "honest_limitation" | "harmful",
  "goal_progress": "achieved" | "partial" | "none",
  "new_state": "seeking" | "clarifying" | "frustrated" | "escalating" | "satisfied" | "abandoned",
  "reasoning": "<1 sentence: why this response type and state>",
  "next_message": "<persona's next message in their voice, or null if goal achieved or abandoned>"
}}

Rules:
- If goal_progress is "achieved" → new_state must be "satisfied", next_message must be null
- If persona is frustrated and response is vague again → escalate or abandon based on patience
- next_message must match the persona's current emotional state (frustrated → more blunt)
- If next_message is null, the conversation ends
"""


def _get_adapter(config: RunConfig) -> object:
    """Select the right adapter based on application type."""
    kwargs = dict(
        endpoint_url=config.endpoint_url,
        request_field=config.request_field,
        response_field=config.response_field,
        auth_type=config.auth_type,
        auth_token=config.auth_token,
    )
    if config.application_type == ApplicationType.RAG:
        return RAGAdapter(**kwargs)
    elif config.application_type == ApplicationType.MULTI_AGENT:
        return MultiAgentAdapter(**kwargs)
    elif config.application_type == ApplicationType.FORM:
        return FormAdapter(**kwargs)
    else:
        return ChatbotAdapter(**kwargs)


async def run_conversation(
    persona: Persona,
    prompt: GeneratedPrompt,
    config: RunConfig,
    llm_client: LLMClient,
    project_id: str = "",
) -> Conversation:
    """
    Run a full conversation for one persona + prompt pair.
    Returns a Conversation with all turns logged.
    """
    adapter = _get_adapter(config)
    conversation = Conversation(
        project_id=project_id,
        persona_id=persona.persona_id,
        persona_name=persona.name,
        test_class=prompt.test_class,
        attack_category=prompt.attack_category,
        started_at=datetime.utcnow(),
    )

    current_state = ConversationState.SEEKING
    current_message = prompt.text   # Start with the generated prompt
    history_lines: List[str] = []

    for turn_num in range(1, MAX_TURNS + 1):
        # Send to adapter
        resp: AdapterResponse = await adapter.send(current_message)
        latency_ms = resp.latency_ms

        # Detect error response — includes field-not-found and HTTP errors
        is_error = (
            not resp.ok
            or not resp.text
            or resp.text.startswith("[Error")
            or resp.text.startswith("[Field")
            or resp.text.startswith("[Index")
            or resp.text.startswith("[Empty")
        )

        # Build turn record — carry expected_behavior from prompt (turn 1 only; turns 2+ don't have it)
        turn = ConversationTurn(
            turn_number=turn_num,
            query=current_message,
            response=resp.text if resp.ok else f"[Error: {resp.error}]",
            latency_ms=latency_ms,
            expected_behavior=prompt.expected_behavior if turn_num == 1 else None,
            retrieved_context=resp.retrieved_context,
            agent_trace=resp.agent_trace,
            persona_state=current_state,
            is_error_response=is_error,
            timestamp=datetime.utcnow(),
        )
        conversation.turns.append(turn)
        conversation.total_latency_ms += latency_ms

        history_lines.append(f"User: {current_message}")
        history_lines.append(f"AI: {turn.response[:300]}")

        # Stop early: last turn reached, or chatbot returned an error/bad field
        if turn_num >= MAX_TURNS or is_error:
            if is_error:
                conversation.goal_achieved = False
            break

        # Combined eval+generate for subsequent turns
        next_msg, new_state, response_type, goal_achieved = await _combined_eval_generate(
            persona=persona,
            ai_response=turn.response,
            history="\n".join(history_lines[-8:]),   # last 4 exchanges
            current_state=current_state,
            llm_client=llm_client,
        )

        # Update turn with evaluation
        turn.persona_state = new_state
        turn.response_type = response_type

        current_state = new_state

        if goal_achieved or next_msg is None or new_state in (
            ConversationState.SATISFIED, ConversationState.ABANDONED
        ):
            conversation.goal_achieved = goal_achieved
            break

        current_message = next_msg

    conversation.final_state = current_state
    if conversation.goal_achieved is None:
        conversation.goal_achieved = (current_state == ConversationState.SATISFIED)
    conversation.ended_at = datetime.utcnow()
    return conversation


async def _combined_eval_generate(
    persona: Persona,
    ai_response: str,
    history: str,
    current_state: ConversationState,
    llm_client: LLMClient,
) -> tuple[Optional[str], ConversationState, Optional[ResponseType], bool]:
    """
    Single LLM call: evaluate AI response + generate persona's next message.
    Returns (next_message, new_state, response_type, goal_achieved).
    """
    prompt = COMBINED_TURN_PROMPT.format(
        name=persona.name,
        goal=persona.goal,
        current_state=current_state.value,
        patience=persona.behavioral_params.patience_level,
        persistence=persona.behavioral_params.persistence,
        escalation_trigger=persona.behavioral_params.escalation_trigger,
        abandon_trigger=persona.behavioral_params.abandon_trigger,
        base_style=persona.language_model.base_style or "conversational",
        frustrated_style=persona.language_model.frustrated_style or "more direct",
        response=ai_response[:800],
        history=history,
    )

    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.3, max_tokens=600, task="fast", retries=2,
        )
        goal_progress = data.get("goal_progress", "none")
        goal_achieved = goal_progress == "achieved"

        new_state_raw = data.get("new_state", "seeking")
        try:
            new_state = ConversationState(new_state_raw)
        except ValueError:
            new_state = ConversationState.SEEKING

        resp_type_raw = data.get("response_type", "vague")
        try:
            response_type = ResponseType(resp_type_raw)
        except ValueError:
            response_type = ResponseType.VAGUE

        next_msg = data.get("next_message")
        if next_msg and len(str(next_msg).strip()) < 3:
            next_msg = None

        return next_msg, new_state, response_type, goal_achieved
    except Exception:
        # Fallback: assume partial progress, keep going
        return None, ConversationState.SATISFIED, ResponseType.VAGUE, False


async def run_all_conversations(
    personas: List[Persona],
    prompts: List[GeneratedPrompt],
    config: RunConfig,
    llm_client: LLMClient,
    project_id: str = "",
    progress_callback=None,
) -> List[Conversation]:
    """
    Run conversations for all persona+prompt pairs.
    Groups prompts by persona_id for efficient processing.
    """
    # Build persona map
    persona_map = {p.persona_id: p for p in personas}

    # Group prompts by test class (functional first, then security, then performance)
    ordered = sorted(prompts, key=lambda p: (
        0 if p.test_class == TestClass.FUNCTIONAL else
        1 if p.test_class == TestClass.SECURITY else
        2
    ))

    total = len(ordered)
    completed = 0
    sem = asyncio.Semaphore(5)  # max 3 concurrent conversations

    async def _run_one(prompt):
        nonlocal completed
        persona = persona_map.get(prompt.persona_id)
        if not persona:
            return None
        async with sem:
            conv = await run_conversation(persona, prompt, config, llm_client, project_id)
        completed += 1
        if progress_callback:
            progress_callback(completed, total, conv)
        return conv

    raw = await asyncio.gather(*[_run_one(p) for p in ordered])
    return [c for c in raw if c is not None]
