"""
Stage 0: App Profile extraction.
Parses a seed document (plain text describing the AI app) into a structured
AppProfile using LLM. If document is empty, constructs AppProfile from the
manual configure-form fields (fallback path).

Sourced from new metron-backend/app/stage1_understand/document_parser.py.
"""

from __future__ import annotations
import json

from core.llm_client import LLMClient
from core.models import AppProfile, ApplicationType, AgentDefinition

SYSTEM_PROMPT = """You are an expert AI systems analyst.
Extract structured metadata from a description of an AI application.
Return ONLY valid JSON. No markdown, no explanation."""

EXTRACTION_PROMPT = """
Analyze this AI application description and extract structured metadata.

DOCUMENT:
{document}

Return JSON with exactly these fields:
{{
  "application_type": "chatbot" | "rag" | "multi_agent" | "form",
  "domain": "<single word: finance | medical | travel | legal | ecommerce | education | hr | general | ...>",
  "user_types": ["<who uses this app, 2-4 types>"],
  "use_cases": ["<what they use it for, 3-5 cases>"],
  "domain_vocabulary": ["<domain-specific terms the users know, 5-10 words>"],
  "boundaries": ["<what the app explicitly cannot or should not do, 2-4 items>"],
  "success_criteria": {{
    "<use_case_1>": "<what a good response looks like>",
    "<use_case_2>": "<what a good response looks like>"
  }},
  "agents": [
    {{"name": "<agent name if multi-agent>", "role": "<role>", "capabilities": ["<capability>"]}}
  ]
}}

Rules:
- application_type: use "rag" if the app retrieves documents, "multi_agent" if multiple
  specialized agents collaborate, "form" if it processes form-encoded input, else "chatbot"
- domain: pick the single most relevant domain word
- agents: empty list for chatbot/rag/form types; 1-3 entries for multi_agent
- Keep all arrays non-empty (minimum 1 item each)
"""


async def parse_document(
    document_text: str,
    llm_client: LLMClient,
    project_id: str = "",
) -> AppProfile:
    """Extract AppProfile from seed document using LLM."""
    prompt = EXTRACTION_PROMPT.format(document=document_text[:4000])
    try:
        data = await llm_client.complete_json(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=2000,
            retries=4,
        )
    except Exception as e:
        # Fallback: minimal profile
        return AppProfile(
            project_id=project_id,
            application_type=ApplicationType.CHATBOT,
            user_types=["general user"],
            use_cases=["general assistance"],
            domain="general",
            raw_document=document_text,
        )

    agents = [
        AgentDefinition(
            name=a.get("name", "Agent"),
            role=a.get("role", ""),
            capabilities=a.get("capabilities", []),
        )
        for a in data.get("agents", [])
    ]

    app_type_raw = data.get("application_type", "chatbot").lower()
    try:
        app_type = ApplicationType(app_type_raw)
    except ValueError:
        app_type = ApplicationType.CHATBOT

    return AppProfile(
        project_id=project_id,
        application_type=app_type,
        agents=agents,
        user_types=data.get("user_types", ["general user"]) or ["general user"],
        use_cases=data.get("use_cases", ["general assistance"]) or ["general assistance"],
        domain=data.get("domain", "general") or "general",
        domain_vocabulary=data.get("domain_vocabulary", []),
        boundaries=data.get("boundaries", []),
        success_criteria=data.get("success_criteria", {}),
        raw_document=document_text,
    )


def build_profile_from_config(
    agent_description: str,
    agent_domain: str,
    application_type_str: str,
    is_rag: bool = False,
    project_id: str = "",
) -> AppProfile:
    """
    Construct AppProfile from the configure-form fields (no LLM needed).
    Used when user fills the form manually without uploading a seed document.
    """
    # Resolve application type
    if is_rag:
        app_type = ApplicationType.RAG
    else:
        mapping = {
            "rag": ApplicationType.RAG,
            "multi_agent": ApplicationType.MULTI_AGENT,
            "multiagent": ApplicationType.MULTI_AGENT,
            "form": ApplicationType.FORM,
        }
        app_type = mapping.get(application_type_str.lower(), ApplicationType.CHATBOT)

    return AppProfile(
        project_id=project_id,
        application_type=app_type,
        user_types=["general user", "power user", "novice user"],
        use_cases=[agent_description[:200]] if agent_description else ["general assistance"],
        domain=agent_domain.lower() or "general",
        domain_vocabulary=[],
        boundaries=[],
        success_criteria={},
        raw_document=agent_description,
    )
