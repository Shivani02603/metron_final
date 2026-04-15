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
from core import db as _db
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

# In-memory job store — keyed by run_id
# Fix 22: stores only status/progress/results (NO llm_api_key or auth_token)
jobs: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def _startup():
    """Fix 21+28: init DB and re-populate in-memory jobs from recent completed runs."""
    try:
        _db.init_db()
        for row in _db.load_recent_jobs(hours=24):
            run_id = row["run_id"]
            jobs[run_id] = {
                "status":        row["status"],
                "progress":      100,
                "message":       "Completed (recovered from DB)",
                "current_phase": "",
                "phase_results": {},
                "log_events":    [],
                "error":         None,
                "results":       row.get("results"),
            }
        print(f"[DB] Recovered {len(jobs)} recent runs from SQLite on startup.")
    except Exception as e:
        print(f"[DB] Startup recovery failed (non-fatal): {e}")

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

    def _check(pkg: str, attr: str = "") -> bool:
        try:
            mod = __import__(pkg)
            if attr:
                return hasattr(mod, attr) or True  # sub-import check
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _check_sub(pkg: str, subpath: str) -> bool:
        """Check a submodule import like 'deepeval.metrics.GEval'."""
        try:
            parts = subpath.split(".")
            mod = __import__(pkg)
            for part in parts:
                mod = getattr(mod, part, None)
                if mod is None:
                    return False
            return True
        except Exception:
            return False

    # PII detection
    presidio_ok = _check("presidio_analyzer")
    tools["presidio"] = {
        "installed": presidio_ok,
        "description": "PII detection (Presidio — replaces LLM PII guessing)",
        "used_for": "pii_leakage metric in security evaluation",
    }

    # Toxicity classifier
    detoxify_ok = _check("detoxify")
    tools["detoxify"] = {
        "installed": detoxify_ok,
        "description": "Toxicity classifier (Detoxify — replaces LLM toxicity scoring)",
        "used_for": "toxicity metric in security evaluation",
    }

    # Prompt injection scanner
    llmguard_ok = _check("llm_guard")
    tools["llm_guard"] = {
        "installed": llmguard_ok,
        "description": "Prompt injection scanner (LLM Guard — replaces LLM injection guessing)",
        "used_for": "prompt_injection metric in security evaluation",
    }

    # DeepEval
    deepeval_ok = _check("deepeval")
    tools["deepeval"] = {
        "installed": deepeval_ok,
        "description": "Structured LLM evaluation (DeepEval — GEval, Hallucination, Bias, Relevancy)",
        "used_for": "hallucination + answer_relevancy in functional; geval in quality; bias in security",
    }

    # RAGAS
    ragas_ok = _check("ragas")
    tools["ragas"] = {
        "installed": ragas_ok,
        "description": "RAG evaluation framework (RAGAS — structural faithfulness, no LLM)",
        "used_for": "ragas_faithfulness in quality evaluation (RAG mode only)",
    }

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
    ground_truth_file: Optional[UploadFile] = File(None),
):
    import json, csv, io
    try:
        config_data = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid config JSON")

    # Parse ground truth file (CSV or JSON) into list of {question, expected_answer, context}
    if ground_truth_file:
        try:
            raw = (await ground_truth_file.read()).decode("utf-8", errors="ignore")
            filename = ground_truth_file.filename or ""
            if filename.endswith(".json"):
                parsed = json.loads(raw)

                # ── Locate the list of Q&A pairs regardless of JSON shape ──────
                # Supported shapes:
                #   1. Root array:          [{"question": ...}, ...]
                #   2. Root object/wrapper: {"test_cases": [...]} or
                #                          {"data": [...]} or
                #                          {"questions": [...]} etc.
                rows = None
                if isinstance(parsed, list):
                    rows = parsed
                elif isinstance(parsed, dict):
                    # Recursive search for a list of Q&A pairs
                    def find_list(obj):
                        if isinstance(obj, list):
                            return obj
                        if isinstance(obj, dict):
                            for wrapper_key in (
                                "test_cases", "cases", "questions", "data",
                                "items", "entries", "records", "samples",
                                "ground_truth", "qa_pairs", "pairs",
                            ):
                                if wrapper_key in obj and isinstance(obj[wrapper_key], list):
                                    return obj[wrapper_key]
                            for v in obj.values():
                                found = find_list(v)
                                if found:
                                    return found
                        return None
                    rows = find_list(parsed)

                if rows:
                    pairs = []
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        # Question — accept multiple field names
                        q = (
                            r.get("question") or r.get("query") or
                            r.get("q") or r.get("input") or
                            r.get("user_input") or r.get("prompt") or ""
                        )
                        # Expected answer — accept multiple field names
                        a = (
                            r.get("expected_answer") or r.get("answer") or
                            r.get("a") or r.get("reference") or
                            r.get("expected_output") or r.get("ground_truth") or ""
                        )
                        # Context — accept multiple field names; preserve list as-is
                        c = (
                            r.get("context") or r.get("expected_chunk") or
                            r.get("chunk") or r.get("contexts") or
                            r.get("retrieved_context") or r.get("source") or
                            r.get("passages") or ""
                        )
                        if q and a:
                            pairs.append({
                                "question":        str(q).strip(),
                                "expected_answer": str(a).strip(),
                                # Preserve list context as-is so _ground_truth_to_prompts
                                # can keep individual chunks separate.
                                "context": c,
                            })
                    config_data["ground_truth"] = pairs
                    print(f"[API] Parsed {len(pairs)} ground truth pairs from JSON ({filename})")
                else:
                    print(f"[API] Could not locate a list of Q&A pairs in JSON file: {filename}")

            else:
                # CSV: flexible header — map common column name variants
                reader = csv.DictReader(io.StringIO(raw))
                pairs = []
                for row in reader:
                    q = (
                        row.get("question") or row.get("query") or
                        row.get("q") or row.get("input") or
                        row.get("user_input") or ""
                    )
                    a = (
                        row.get("expected_answer") or row.get("answer") or
                        row.get("a") or row.get("reference") or
                        row.get("ground_truth") or ""
                    )
                    c = (
                        row.get("context") or row.get("expected_chunk") or
                        row.get("chunk") or row.get("contexts") or
                        row.get("source") or ""
                    )
                    if q and a:
                        pairs.append({
                            "question":        q.strip(),
                            "expected_answer": a.strip(),
                            "context":         c.strip(),
                        })
                config_data["ground_truth"] = pairs
                print(f"[API] Parsed {len(pairs)} ground truth pairs from CSV ({filename})")

        except Exception as e:
            print(f"[API] Could not parse ground truth file: {e}")

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

    # Fix 29: project_id comes from config (set by UI from dashboard [id]) or defaults to run_id
    project_id = run_config.project_id or run_id

    # Fix 22: job store contains NO API keys — only status/progress/config summary
    jobs[run_id] = {
        "status":        "queued",
        "progress":      0,
        "message":       "Queued",
        "current_phase": "",
        "phase_results": {},
        "log_events":    [],
        "error":         None,
        "results":       None,
        # Safe config summary (no credentials)
        "config_summary": {
            "endpoint_url":    run_config.endpoint_url,
            "agent_domain":    run_config.agent_domain,
            "llm_provider":    run_config.llm_provider,
            "application_type": run_config.application_type.value,
        },
    }

    background_tasks.add_task(
        run_pipeline,
        run_id=run_id,
        config=run_config,
        job_store=jobs,
        doc_text=doc_text,
        project_id=project_id,
    )

    return {"run_id": run_id, "project_id": project_id}


# ──────────────────────────────────────────────────────────────────────────
# GET /api/job/{run_id}/status — poll job status
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/job/{run_id}/status")
async def get_job_status(run_id: str):
    job = jobs.get(run_id)
    if not job:
        # Fix 21: fall back to DB for runs that completed before last restart
        db_row = _db.get_run(run_id)
        if db_row:
            return {
                "run_id":        run_id,
                "status":        db_row.get("status", "completed"),
                "progress":      100,
                "message":       "Completed (from DB)",
                "current_phase": "",
                "phase_results": {},
                "log_events":    [],
                "error":         None,
            }
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
        # Fix 21: fall back to DB
        db_row = _db.get_run(run_id)
        if db_row and db_row.get("results"):
            return db_row["results"]
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


# ──────────────────────────────────────────────────────────────────────────
# GET /api/job/{run_id}/status — also checks DB if not in memory (Fix 21)
# (Replaces the original endpoint above with DB fallback)
# ──────────────────────────────────────────────────────────────────────────

# NOTE: The original GET /api/job/{run_id}/status endpoint stays at line 393+
# as-is. We add DB fallback there via an override at import time.
# Actually we patch it here:

# ──────────────────────────────────────────────────────────────────────────
# GET /api/project/{project_id}/runs — Fix 28: run history for a project
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/project/{project_id}/runs")
async def get_project_runs(project_id: str):
    """Return all completed runs for a project, sorted newest-first."""
    try:
        runs = _db.get_runs_for_project(project_id)
        return {"project_id": project_id, "runs": runs}
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# GET /api/runs/{run_id_a}/compare/{run_id_b} — Fix 28: regression diff
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/runs/{run_id_a}/compare/{run_id_b}")
async def compare_runs(run_id_a: str, run_id_b: str):
    """Compare health scores and class pass-rates between two runs."""
    try:
        diff = _db.compare_runs(run_id_a, run_id_b)
        if "error" in diff:
            raise HTTPException(404, diff["error"])
        return diff
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Compare error: {e}")


# ── Helper ─────────────────────────────────────────────────────────────────
def _env_key_set(provider_name: str) -> bool:
    env_key = LLM_PROVIDERS.get(provider_name, {}).get("env_key", "")
    return bool(env_key and os.environ.get(env_key))
