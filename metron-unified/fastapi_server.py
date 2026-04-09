"""
FastAPI server — unified METRON backend.
8 endpoints: all 7 from existing backend (backward-compatible) + new /api/parse-document.
Runs pipeline.py as a background task, stores jobs in-memory.
"""

from __future__ import annotations
import asyncio
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.config import CORS_ORIGINS, LLM_PROVIDERS
from core.llm_client import LLMClient
from core.models import (
    ApplicationType, ConnectTestRequest, JobStatus,
    ParseDocumentRequest, PreviewRequest, RunConfig,
)
from core.adapters.chatbot import ChatbotAdapter
from pipeline import run_pipeline
from stages.s0_profile.document_parser import parse_document
from stages.s1_personas.fishbone_builder import build_slots
from stages.s1_personas.persona_builder import build_all_personas

app = FastAPI(title="METRON Unified API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
jobs: Dict[str, Dict[str, Any]] = {}

# ──────────────────────────────────────────────────────────────────────────
# GET /api/providers — list LLM providers
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/providers")
async def get_providers():
    return {
        name: {
            "description": info["description"],
            "rpm":         info["rpm"],
            "models":      info["models"],
            "default":     info["default"],
            "env_key":     info["env_key"],
            "token_optimize": info.get("token_optimize", False),
        }
        for name, info in LLM_PROVIDERS.items()
    }


# ──────────────────────────────────────────────────────────────────────────
# GET /api/tools/status — check optional tool availability
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/tools/status")
async def get_tools_status():
    tools = {}

    def _check(pkg: str) -> bool:
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False

    tools["garak"]      = {"installed": _check("garak"),     "description": "LLM vulnerability scanner"}
    tools["ragas"]      = {"installed": _check("ragas"),      "description": "RAG evaluation metrics"}
    tools["deepeval"]   = {"installed": _check("deepeval"),   "description": "LLM evaluation framework"}
    tools["rouge_score"] = {"installed": _check("rouge_score"), "description": "ROUGE text similarity"}
    tools["bert_score"] = {"installed": _check("bert_score"), "description": "BERTScore semantic similarity"}
    tools["neo4j"]      = {"installed": _check("neo4j"),      "description": "Persona registry (optional)"}
    return tools


# ──────────────────────────────────────────────────────────────────────────
# POST /api/connect-test — test chatbot endpoint connectivity
# ──────────────────────────────────────────────────────────────────────────
@app.post("/api/connect-test")
async def connect_test(req: ConnectTestRequest):
    adapter = ChatbotAdapter(
        endpoint_url=req.endpoint_url,
        request_field=req.request_field,
        response_field=req.response_field,
        auth_type=req.auth_type,
        auth_token=req.auth_token,
    )
    success, message = await adapter.test_connection()
    return {"success": success, "message": message}


# ──────────────────────────────────────────────────────────────────────────
# POST /api/parse-document — NEW: seed doc → AppProfile
# ──────────────────────────────────────────────────────────────────────────
@app.post("/api/parse-document")
async def parse_document_endpoint(req: ParseDocumentRequest):
    if not req.document_text.strip():
        raise HTTPException(400, "document_text is required")
    if not req.llm_api_key and not _env_key_set(req.llm_provider):
        raise HTTPException(400, f"API key required for {req.llm_provider}")

    llm_client = LLMClient(req.llm_provider, req.llm_api_key)
    profile = await parse_document(req.document_text, llm_client)
    return {
        "application_type":  profile.application_type.value,
        "domain":            profile.domain,
        "user_types":        profile.user_types,
        "use_cases":         profile.use_cases,
        "domain_vocabulary": profile.domain_vocabulary,
        "boundaries":        profile.boundaries,
        "success_criteria":  profile.success_criteria,
        "agents":            [a.model_dump() for a in profile.agents],
    }


# ──────────────────────────────────────────────────────────────────────────
# POST /api/preview — generate personas + scenarios
# ──────────────────────────────────────────────────────────────────────────
@app.post("/api/preview")
async def preview(req: PreviewRequest):
    if not req.agent_description.strip():
        raise HTTPException(400, "agent_description is required")
    if not req.llm_api_key and not _env_key_set(req.llm_provider):
        raise HTTPException(400, f"API key required for {req.llm_provider}")

    llm_client = LLMClient(req.llm_provider, req.llm_api_key)

    from stages.s0_profile.document_parser import build_profile_from_config
    profile = build_profile_from_config(
        agent_description=req.agent_description,
        agent_domain=req.agent_domain,
        application_type_str=req.application_type,
    )

    # Fishbone slots → personas
    slots    = build_slots(profile, num_personas=req.num_personas)
    personas = await build_all_personas(slots, profile, llm_client)

    # Generate scenarios (functional prompts as scenarios for UI compatibility)
    from stages.s2_tests.functional_gen import generate_functional_prompts
    scenarios = []
    for p in personas[:req.num_scenarios]:
        prompts = await generate_functional_prompts(p, profile, llm_client)
        for pr in prompts[:2]:
            scenarios.append({
                "id":               pr.prompt_id,
                "name":             f"{p.name} — {p.goal[:40]}",
                "description":      f"Test for {p.expertise.value} {p.user_type}",
                "initial_prompt":   pr.text,
                "expected_behavior": pr.expected_behavior or "",
                "category":         "functional",
            })

    return {
        "personas": [
            {
                "id":             p.persona_id,
                "name":           p.name,
                "description":    p.description or p.background[:200],
                "traits":         p.traits,
                "sample_prompts": p.sample_prompts or p.entry_points,
                "fishbone":       p.fishbone_dimensions,
                "expertise":      p.expertise.value,
                "emotional_state": p.emotional_state.value,
                "intent":         p.intent.value,
            }
            for p in personas
        ],
        "scenarios": scenarios,
    }


# ──────────────────────────────────────────────────────────────────────────
# POST /api/run — submit test job
# ──────────────────────────────────────────────────────────────────────────
@app.post("/api/run")
async def run_tests(
    background_tasks: BackgroundTasks,
    config: str = Form(...),
    document: Optional[UploadFile] = File(None),
):
    import json
    try:
        config_data = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid config JSON")

    run_config = RunConfig(**config_data)

    if not run_config.llm_api_key and not _env_key_set(run_config.llm_provider):
        raise HTTPException(400, f"API key required for {run_config.llm_provider}")

    # Read uploaded document
    doc_text = ""
    if document:
        try:
            content = await document.read()
            doc_text = content.decode("utf-8", errors="ignore")
        except Exception:
            doc_text = ""

    run_id = str(uuid.uuid4())
    jobs[run_id] = {
        "status":       "queued",
        "progress":     0,
        "message":      "Queued",
        "current_phase": "",
        "phase_results": {},
        "log_events":   [],
        "error":        None,
        "results":      None,
    }

    background_tasks.add_task(
        run_pipeline,
        run_id=run_id,
        config=run_config,
        job_store=jobs,
        doc_text=doc_text,
        project_id=run_id,
    )

    return {"run_id": run_id}


# ──────────────────────────────────────────────────────────────────────────
# GET /api/job/{run_id}/status — poll job status
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/job/{run_id}/status")
async def get_job_status(run_id: str):
    job = jobs.get(run_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "run_id":        run_id,
        "status":        job["status"],
        "progress":      job["progress"],
        "message":       job["message"],
        "current_phase": job["current_phase"],
        "phase_results": job["phase_results"],
        "log_events":    job.get("log_events", []),
        "error":         job["error"],
    }


# ──────────────────────────────────────────────────────────────────────────
# GET /api/job/{run_id}/results — fetch completed results
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/job/{run_id}/results")
async def get_job_results(run_id: str):
    job = jobs.get(run_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "running" or job["status"] == "queued":
        raise HTTPException(202, "Job still running")
    if job["status"] == "failed":
        raise HTTPException(500, job.get("error", "Pipeline failed"))
    return job["results"]


# ──────────────────────────────────────────────────────────────────────────
# GET /api/health
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Helper ─────────────────────────────────────────────────────────────────
def _env_key_set(provider_name: str) -> bool:
    env_key = LLM_PROVIDERS.get(provider_name, {}).get("env_key", "")
    return bool(env_key and os.environ.get(env_key))
