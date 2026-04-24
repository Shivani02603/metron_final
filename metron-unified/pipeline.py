"""
Master 8-stage pipeline orchestrator.
Connects all stages, updates job_store progress at each stage,
and handles the full lifecycle from AppProfile → AggregatedReport.
"""

from __future__ import annotations
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Per-run locks prevent concurrent pipeline stages from overwriting each other's
# job_store state when multiple runs execute simultaneously.
_job_locks: Dict[str, asyncio.Lock] = {}


def _get_lock(run_id: str) -> asyncio.Lock:
    if run_id not in _job_locks:
        _job_locks[run_id] = asyncio.Lock()
    return _job_locks[run_id]

from core.llm_client import LLMClient
from core.models import (
    AppProfile, AggregatedReport, ApplicationType, Conversation, GeneratedPrompt,
    MetricResult, Persona, RunConfig,
)

# Stage imports
from stages.s0_profile.document_parser import parse_document, build_profile_from_config
from stages.s0_profile.technical_extractor import extract_technical_profile
from core.attack_surface_mapper import map_attack_surface
from stages.s1_personas.fishbone_builder import build_slots
from stages.s1_personas.persona_builder import build_all_personas, build_persona
from stages.s1_personas.coverage_validator import validate_and_fill
from stages.s2_tests.functional_gen import generate_all_functional, generate_performance_prompts
from stages.s2_tests.security_gen import generate_all_security
from stages.s2_tests.quality_criteria import generate_quality_criteria
from stages.s3_execution.conversation_runner import run_all_conversations, run_ground_truth_conversations
from stages.s4_evaluation.functional import evaluate_functional
from stages.s4_evaluation.security import evaluate_security
from stages.s4_evaluation.quality import evaluate_quality
from stages.s4_evaluation.performance import evaluate_performance
from stages.s4_evaluation.load import evaluate_load
from stages.s4_evaluation.rag import evaluate_rag
from stages.s5_aggregation.aggregator import aggregate
from stages.s7_report.report_generator import report_to_json, generate_html_report
from stages.s8_rca.rca_mapper import run_rca
from stages.s8_rca.prompt_classifier import classify_prompt_failures


# ── Progress helpers ───────────────────────────────────────────────────────

def _log(job_store: Dict, run_id: str, event_type: str, content: Dict):
    """Append a rich log event to the job's event stream for the live feed UI."""
    if run_id not in job_store:
        return
    job_store[run_id].setdefault("log_events", []).append({
        "type": event_type,
        "ts": datetime.utcnow().strftime("%H:%M:%S"),
        "content": content,
    })


def _update(job_store: Dict, run_id: str, progress: int, message: str, phase: str = "", phase_data: Optional[Dict] = None):
    if run_id in job_store:
        job_store[run_id]["progress"]      = progress
        job_store[run_id]["message"]       = message
        job_store[run_id]["current_phase"] = phase
        if phase_data is not None:
            job_store[run_id]["phase_results"][phase] = phase_data


# ── Main pipeline entry point ──────────────────────────────────────────────

async def run_pipeline(
    run_id:     str,
    config:     RunConfig,
    job_store:  Dict[str, Any],
    doc_text:   str = "",
    project_id: str = "",
) -> None:
    """
    Full 8-stage pipeline. Updates job_store at each stage.
    Called as a background task from FastAPI.
    """
    job_store[run_id]["status"] = "running"
    llm_client = LLMClient(config.llm_provider, config.llm_api_key)

    try:
        # ── Stage 0: App Profile ───────────────────────────────────────────
        _update(job_store, run_id, 5, "Analyzing agent profile…", "profiling")
        if doc_text.strip():
            profile = await parse_document(doc_text, llm_client, project_id)
            # Override app type from config if explicitly set
            if config.application_type.value != "chatbot":
                profile.application_type = config.application_type
            # is_rag=True must always win — the dropdown and the toggle can be
            # out of sync when users toggle RAG without changing application_type.
            if config.is_rag:
                profile.application_type = ApplicationType.RAG
            if config.agent_domain:
                profile.domain = config.agent_domain.lower()
        else:
            profile = build_profile_from_config(
                agent_description=config.agent_description,
                agent_domain=config.agent_domain,
                application_type_str=config.application_type.value,
                is_rag=config.is_rag,
                project_id=project_id,
            )

        # ── Stage 0b: Technical Profile Extraction ────────────────────────
        # Second LLM pass: extracts 24-category technical attack surface from
        # the seed document. Used to generate stack-specific red-team probes.
        # Runs only when a document is provided; pipeline continues if it fails.
        tech_profile = None
        attack_vectors = []
        if doc_text.strip():
            _update(job_store, run_id, 7, "Extracting technical attack surface…", "profiling")
            tech_profile = await extract_technical_profile(doc_text, llm_client)
            attack_vectors = map_attack_surface(tech_profile)
            _log(job_store, run_id, "technical_profile", {
                "vectors_found": len(attack_vectors),
                "summary": tech_profile.extraction_summary[:200] if tech_profile.extraction_summary else "",
                "authorized_actions": tech_profile.authorized_actions[:5],
                "output_destinations": tech_profile.output_destinations[:3],
                "compliance_frameworks": tech_profile.compliance_frameworks,
            })

        # ── Stage 1: Persona Generation (Fishbone) ─────────────────────────
        _update(job_store, run_id, 10, "Building persona coverage matrix…", "personas")
        _log(job_store, run_id, "phase_start", {"phase": "personas", "label": "Persona Generation"})
        slots = build_slots(profile, num_personas=config.num_personas)
        personas = await build_all_personas(slots, profile, llm_client, project_id, tech_profile)

        # Coverage validation (adds up to 3 more slots if gaps found)
        extra_slots = await validate_and_fill(personas, profile, llm_client)
        if extra_slots:
            extra_personas = await build_all_personas(extra_slots, profile, llm_client, project_id, tech_profile)
            personas.extend(extra_personas)

        # Emit one event per persona
        for p in personas:
            _log(job_store, run_id, "persona_created", {
                "name": p.name,
                "intent": p.intent.value,
                "expertise": p.expertise.value,
                "emotional_state": p.emotional_state.value,
                "goal": p.goal[:120] if p.goal else "",
                "user_type": p.user_type,
            })

        _update(job_store, run_id, 18, f"Generated {len(personas)} personas", "personas",
                {"count": len(personas), "names": [p.name for p in personas]})

        # ── Stage 2: Domain-Specific Test Generation ───────────────────────
        _update(job_store, run_id, 20, "Generating domain-specific test prompts…", "test_gen")
        _log(job_store, run_id, "phase_start", {"phase": "test_gen", "label": "Test Generation"})

        # 2a: Functional prompts — always LLM-generated, grounded in rag_text for RAG mode.
        # Ground truth Q&A pairs are handled separately in Stream 2 (after Stage 3).
        rag_text = config.rag_text if config.is_rag else ""
        func_prompts = await generate_all_functional(
            personas, profile, llm_client,
            rag_text=rag_text,
            max_prompts=config.num_scenarios or 0,   # Fix 35: wire UI num_scenarios cap
        )

        # 2b: Security prompts (adversarial personas only) + technical probes
        sec_prompts = await generate_all_security(
            personas, profile, llm_client,
            selected_categories=config.selected_attacks,
            attacks_per_category=config.attacks_per_category,
            attack_vectors=attack_vectors,
            tech_profile=tech_profile,
        )

        # 2c: Quality criteria
        quality_criteria = await generate_quality_criteria(profile, llm_client)

        all_prompts = func_prompts + sec_prompts

        # Log sample prompts (first 3 functional, first 2 security)
        for fp in func_prompts[:3]:
            persona_name = next((p.name for p in personas if p.persona_id == fp.persona_id), "Unknown")
            _log(job_store, run_id, "test_prompt", {
                "test_class": "functional",
                "persona_name": persona_name,
                "text": fp.text[:200],
                "expected_behavior": (fp.expected_behavior or "")[:120],
            })
        for sp in sec_prompts[:2]:
            persona_name = next((p.name for p in personas if p.persona_id == sp.persona_id), "Unknown")
            _log(job_store, run_id, "test_prompt", {
                "test_class": "security",
                "persona_name": persona_name,
                "text": sp.text[:200],
                "attack_category": sp.attack_category or "",
                "severity": sp.severity or "medium",
            })

        if quality_criteria and quality_criteria.get("criteria"):
            _log(job_store, run_id, "quality_criteria", {
                "domain": profile.domain,
                "criteria": [c.get("name", "") for c in quality_criteria["criteria"][:6]],
            })

        _update(job_store, run_id, 25, f"Generated {len(all_prompts)} test prompts", "test_gen",
                {"functional": len(func_prompts), "security": len(sec_prompts)})

        # ── Stage 3+4: Execution + Evaluation (interleaved) ────────────────
        _update(job_store, run_id, 28, "Running conversations with target AI…", "execution")
        _log(job_store, run_id, "phase_start", {"phase": "execution", "label": "Running Conversations"})

        # Progress callback for live updates — emits conversation feed events
        def on_conv(done: int, total: int, conv: Conversation):
            progress = 28 + int((done / total) * 30)   # 28-58%
            phase = conv.test_class.value
            _update(job_store, run_id, progress,
                    f"Executing {phase} test {done}/{total}…", "execution")
            # Emit every conversation so user sees live Q&A
            last_turn = conv.turns[-1] if conv.turns else None
            _log(job_store, run_id, "conversation", {
                "persona_name": conv.persona_name,
                "test_class": phase,
                "query": last_turn.query if last_turn else "",
                "response": last_turn.response if last_turn else "",
                "latency_ms": round(conv.total_latency_ms),
                "num_turns": len(conv.turns),
                "done": done,
                "total": total,
            })

        conversations = await run_all_conversations(
            personas, all_prompts, config, llm_client, project_id,
            progress_callback=on_conv,
        )

        # Stage 4: All evaluations in parallel
        # Each evaluator runs internally throttled (sem=3 per evaluator, timeouts on every
        # Azure call) so they cannot hang forever. Progress ticks every ~10s so the UI
        # always shows movement even while evaluation is running.
        _update(job_store, run_id, 58, "Evaluating results (functional + security + quality in parallel)…", "functional")
        _log(job_store, run_id, "phase_start", {"phase": "evaluation", "label": "Evaluating Results"})

        async def _run_evals_with_progress():
            """
            Run three evaluators in parallel with a progress-tick loop.

            Two key rules:
            1. asyncio.create_task() requires a *coroutine*, not a Future.
               asyncio.gather() returns a _GatheringFuture — wrapping it in an
               async def gives create_task a proper coroutine to schedule.
            2. Each evaluator is isolated: one crashing does not lose the others.
            """
            async def _safe(coro, label: str) -> list:
                try:
                    return await coro
                except Exception as exc:
                    print(f"[Pipeline] {label} evaluation failed (non-fatal): {exc}")
                    return []

            async def _all_evals():
                return await asyncio.gather(
                    _safe(evaluate_functional(conversations, personas, config, llm_client, quality_criteria), "functional"),
                    _safe(evaluate_security(conversations, personas, config, llm_client),                     "security"),
                    _safe(evaluate_quality(conversations, personas, config, llm_client, quality_criteria),    "quality"),
                )

            eval_task = asyncio.create_task(_all_evals())
            progress = 58
            progress_labels = [
                "Scoring functional conversations…",
                "Running security checks…",
                "Applying quality criteria…",
                "Finalising evaluation scores…",
            ]
            label_idx = 0
            while not eval_task.done():
                await asyncio.sleep(10)
                if not eval_task.done():
                    progress = min(progress + 1, 67)
                    label = progress_labels[min(label_idx, len(progress_labels) - 1)]
                    label_idx += 1
                    _update(job_store, run_id, progress, label, "evaluation")
            return await eval_task

        func_results, sec_results, qual_results = await _run_evals_with_progress()

        # ── Stream 2: Ground truth direct evaluation (RAG mode only) ─────────
        # Sends ground truth questions straight to the RAG endpoint with no persona
        # state machine. Collected conversations are merged with Stream 1 conversations
        # so evaluate_rag scores both together.
        gt_conversations: List[Conversation] = []
        if config.is_rag and config.ground_truth:
            _update(job_store, run_id, 60, f"Running ground truth stream ({len(config.ground_truth)} Q&A pairs)…", "rag")
            _log(job_store, run_id, "phase_start", {"phase": "rag", "label": "Ground Truth Stream"})
            try:
                gt_conversations = await run_ground_truth_conversations(
                    config.ground_truth, config, project_id
                )
                _log(job_store, run_id, "gt_stream_complete", {
                    "total_pairs": len(config.ground_truth),
                    "completed": len(gt_conversations),
                })
            except Exception as gt_err:
                print(f"[Pipeline] Ground truth stream failed: {gt_err}")

        # RAG evaluation (only in RAG mode) — evaluates Stream 1 + Stream 2 conversations
        rag_results: List[MetricResult] = []
        if config.is_rag:
            all_rag_conversations = conversations + gt_conversations
            _update(job_store, run_id, 70, "Running RAG evaluation (faithfulness, recall, precision, answer correctness)…", "rag")
            _log(job_store, run_id, "phase_start", {"phase": "rag", "label": "RAG Evaluation"})
            try:
                async def _safe_rag():
                    try:
                        return await evaluate_rag(all_rag_conversations, personas, config, llm_client)
                    except Exception as exc:
                        print(f"[Pipeline] RAG evaluation failed (non-fatal): {exc}")
                        return []

                rag_task = asyncio.create_task(_safe_rag())
                rag_progress = 70
                rag_labels = [
                    "Scoring answer relevancy…",
                    "Checking answer correctness…",
                    "Running hallucination checks…",
                    "Finalising RAG metrics…",
                ]
                rag_label_idx = 0
                while not rag_task.done():
                    await asyncio.sleep(10)
                    if not rag_task.done():
                        rag_progress = min(rag_progress + 1, 71)
                        label = rag_labels[min(rag_label_idx, len(rag_labels) - 1)]
                        rag_label_idx += 1
                        _update(job_store, run_id, rag_progress, label, "rag")
                rag_results = await rag_task
            except Exception as rag_err:
                print(f"[Pipeline] RAG task setup failed: {rag_err}")
            _update(job_store, run_id, 71, f"RAG: {len(rag_results)} metrics evaluated", "rag",
                    {"count": len(rag_results), "passed": sum(1 for r in rag_results if r.passed)})

        all_metric_results = func_results + sec_results + qual_results + rag_results

        # Emit top failures + passes for each phase (sample of 4 each)
        for phase_label, results in [("functional", func_results), ("security", sec_results), ("quality", qual_results)]:
            if not results:
                continue
            passed_count = sum(1 for r in results if r.passed)
            avg = round(sum(r.score for r in results) / len(results) * 100, 1) if results else 0
            samples = sorted(results, key=lambda r: r.score)[:4]
            _log(job_store, run_id, "eval_batch", {
                "phase": phase_label,
                "total": len(results),
                "passed": passed_count,
                "avg_score": avg,
                "samples": [
                    {
                        "metric_name": r.metric_name.replace("_", " ").title(),
                        "persona_name": r.persona_name,
                        "score": round(r.score * 100),
                        "passed": r.passed,
                        "reason": (r.reason or "")[:120],
                    }
                    for r in samples
                ],
            })

        # Update phase results for live UI polling
        _update(job_store, run_id, 72, "Functional + Security + Quality evaluated",
                "evaluation", {
                    "functional": {"count": len(func_results), "passed": sum(1 for r in func_results if r.passed)},
                    "security":   {"count": len(sec_results),  "passed": sum(1 for r in sec_results  if r.passed)},
                    "quality":    {"count": len(qual_results),  "passed": sum(1 for r in qual_results if r.passed)},
                })

        # Stage 4: Performance evaluation
        _update(job_store, run_id, 75, "Running performance tests…", "performance")
        _log(job_store, run_id, "phase_start", {"phase": "performance", "label": "Performance Tests"})
        perf_metrics = await evaluate_performance(config, run_id=run_id)
        _log(job_store, run_id, "perf_complete", {
            "avg_latency_ms": round(perf_metrics.get("avg_latency_ms", 0)),
            "p95_latency_ms": round(perf_metrics.get("p95_latency_ms", 0)),
            "p99_latency_ms": round(perf_metrics.get("p99_latency_ms", 0)),
            "error_rate": round(perf_metrics.get("error_rate", 0), 1),
            "throughput_rps": round(perf_metrics.get("throughput_rps", 0), 2),
            "total_requests": perf_metrics.get("total_requests", 0),
            "successful": perf_metrics.get("successful", 0),
        })
        _update(job_store, run_id, 82, f"Performance: p95={perf_metrics.get('p95_latency_ms', 0):.0f}ms",
                "performance", perf_metrics)

        # Stage 4: Load evaluation
        _update(job_store, run_id, 84, f"Running load test ({config.load_concurrent_users} concurrent users)…", "load")
        _log(job_store, run_id, "phase_start", {"phase": "load", "label": f"Load Test — {config.load_concurrent_users} concurrent users"})
        try:
            load_metrics = await evaluate_load(config)
        except Exception as load_err:
            print(f"[Pipeline] Load test failed: {load_err}")
            load_metrics = {
                "tool_used": "locust", "concurrent_users": config.load_concurrent_users,
                "total_requests": 0, "successful": 0, "errors": 0,
                "error_rate": 0.0, "avg_latency_ms": 0.0, "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0, "requests_per_second": 0.0,
                "passed": False, "assessment": f"load test error: {str(load_err)[:100]}",
            }
        _log(job_store, run_id, "load_complete", {
            "concurrent_users": load_metrics.get("concurrent_users", 0),
            "total_requests": load_metrics.get("total_requests", 0),
            "successful": load_metrics.get("successful", 0),
            "error_rate": round(load_metrics.get("error_rate", 0), 1),
            "avg_latency_ms": round(load_metrics.get("avg_latency_ms", 0)),
            "p95_latency_ms": round(load_metrics.get("p95_latency_ms", 0)),
            "requests_per_second": round(load_metrics.get("requests_per_second", 0), 2),
            "assessment": load_metrics.get("assessment", ""),
        })
        _update(job_store, run_id, 89, f"Load: {load_metrics.get('requests_per_second', 0):.1f} RPS",
                "load", load_metrics)

        # ── Stage 5: Aggregation ───────────────────────────────────────────
        _update(job_store, run_id, 90, "Aggregating results…", "aggregation")
        report = aggregate(
            metric_results=all_metric_results,
            conversations=conversations + gt_conversations,
            personas=personas,
            config=config,
            performance_metrics=perf_metrics,
            load_metrics=load_metrics,
            run_id=run_id,
            project_id=project_id,
            # Fix 34: agent_name now lives on AggregatedReport (not added post-hoc)
            agent_name=config.agent_name or (profile.domain.capitalize() + " Agent"),
        )

        # ── Stage 8: Root Cause Analysis ──────────────────────────────────
        _update(job_store, run_id, 94, "Running root cause analysis…", "rca")
        _log(job_store, run_id, "phase_start", {"phase": "rca", "label": "Root Cause Analysis"})
        try:
            rca_report = run_rca(
                metric_results=all_metric_results,
                conversations=conversations + gt_conversations,
                config=config,
                perf_metrics=perf_metrics,
                load_metrics=load_metrics,
            )
            report.rca = rca_report
            _log(job_store, run_id, "rca_complete", {
                "total_failed":     rca_report.total_failed,
                "relevant_points":  rca_report.relevant_points,
                "filtered_points":  rca_report.filtered_points,
                "top_cause":        rca_report.top_causes[0].label if rca_report.top_causes else "none",
                "top_probability":  rca_report.top_causes[0].probability if rca_report.top_causes else 0.0,
            })
            _update(job_store, run_id, 96, f"RCA: {len(rca_report.top_causes)} root causes identified", "rca",
                    {"top_cause": rca_report.top_causes[0].label if rca_report.top_causes else "none"})
        except Exception as rca_err:
            import traceback as _tb
            _rca_tb = _tb.format_exc()
            print(f"[Pipeline] RCA failed (non-fatal):\n{_rca_tb}")
            _log(job_store, run_id, "rca_warning", {
                "message": f"Root cause analysis could not complete: {str(rca_err)[:200]}",
                "detail": _rca_tb[-600:],
            })

        # ── Per-prompt failure classification ──────────────────────────────
        _update(job_store, run_id, 96, "Classifying per-prompt failure reasons…", "rca")
        try:
            all_metric_results = await classify_prompt_failures(
                all_metric_results, config, llm_client
            )
            # Re-sync individual phase lists (they share the same objects, but re-assign for safety)
            func_results = [r for r in all_metric_results if r.superset == "functional"]
            sec_results  = [r for r in all_metric_results if r.superset == "security"]
            qual_results = [r for r in all_metric_results if r.superset == "quality"]
            rag_results  = [r for r in all_metric_results if r.superset == "rag"]

            # Patch failure_drill_down (built before classification) with taxonomy fields
            _clf_index = {
                (r.metric_name, r.persona_name, r.prompt[:80]): r
                for r in all_metric_results
                if r.failure_taxonomy_id
            }
            for entry in report.failure_drill_down:
                key = (entry.get("metric_name",""), entry.get("persona_name",""), entry.get("prompt","")[:80])
                matched = _clf_index.get(key)
                if matched:
                    entry["failure_taxonomy_id"]    = matched.failure_taxonomy_id
                    entry["failure_taxonomy_label"] = matched.failure_taxonomy_label
                    entry["failure_reason"]         = matched.failure_reason
        except Exception as clf_err:
            import traceback as _tb2
            _clf_tb = _tb2.format_exc()
            print(f"[Pipeline] Per-prompt classifier failed (non-fatal):\n{_clf_tb}")
            _log(job_store, run_id, "classifier_warning", {
                "message": f"Per-prompt failure classification could not complete: {str(clf_err)[:200]}",
                "detail": _clf_tb[-600:],
            })

        # ── Stage 7: Report Generation ─────────────────────────────────────
        _update(job_store, run_id, 97, "Generating report…", "report")
        _log(job_store, run_id, "phase_start", {"phase": "report", "label": "Generating Report"})
        report.report_html = generate_html_report(report)
        final_json = report_to_json(report)
        _log(job_store, run_id, "pipeline_complete", {
            "health_score": round(report.health_score * 100, 1),
            "passed": report.passed,
            "total_tests": report.total_tests,
            "total_passed": report.total_passed,
            "domain": report.domain,
        })
        _update(job_store, run_id, 99, "Report ready", "report", {"generated": True})

        # ── Flatten test_classes to top-level for UI compatibility ───────────
        tc = final_json.get("test_classes", {})

        def _to_test_result(r: MetricResult) -> dict:
            return {
                "test_id":    f"{r.metric_name}_{r.persona_id[:8]}_{r.conversation_id}_{r.turn_number or 0}",
                "test_name":  r.persona_name,
                "category":   r.metric_name,   # full name — frontend maps to pretty label
                "input_text": r.prompt,
                "output_text": r.response,
                "score":      r.score,
                "passed":     r.passed,
                "reasoning":  r.reason,
                "latency_ms": r.latency_ms or 0.0,
                "failure_taxonomy_id":    r.failure_taxonomy_id,
                "failure_taxonomy_label": r.failure_taxonomy_label,
                "failure_reason":         r.failure_reason,
                "details": {
                    "severity":            r.severity,
                    "owasp_category":      r.owasp_category,
                    "vulnerability_found": r.vulnerability_found,
                    "pii_detected":        r.pii_detected,
                },
            }



        def _flat_phase(cls_key: str, results: list) -> dict:
            summary = tc.get(cls_key, {})
            return {
                **summary,
                "pass_rate": round(summary.get("pass_rate", 0) * 100, 1),
                "results": [_to_test_result(r) for r in results],
            }

        final_json["functional"] = _flat_phase("functional", func_results)
        final_json["security"]   = _flat_phase("security",   sec_results)
        final_json["quality"]    = _flat_phase("quality",    qual_results)
        if rag_results:
            rag_total  = len(rag_results)
            rag_passed = sum(1 for r in rag_results if r.passed)
            rag_avg    = sum(r.score for r in rag_results) / rag_total if rag_total else 0.0
            final_json["rag"] = {
                "total":     rag_total,
                "passed":    rag_passed,
                "failed":    rag_total - rag_passed,
                "pass_rate": round(rag_passed / rag_total * 100, 1) if rag_total else 0.0,
                "avg_score": round(rag_avg, 4),
                "results":   [_to_test_result(r) for r in rag_results],
            }

        # Normalize performance field names for frontend (removes _ms suffix)
        final_json["performance"] = {
            "total_requests": perf_metrics.get("total_requests", 0),
            "successful":     perf_metrics.get("successful", 0),
            "errors":         perf_metrics.get("errors", 0),
            "error_rate":     perf_metrics.get("error_rate", 0),
            "avg_latency":    perf_metrics.get("avg_latency_ms", 0),
            "min_latency":    perf_metrics.get("min_latency_ms", 0),
            "max_latency":    perf_metrics.get("max_latency_ms", 0),
            "median_latency": perf_metrics.get("median_latency_ms", 0),
            "p95_latency":    perf_metrics.get("p95_latency_ms", 0),
            "p99_latency":    perf_metrics.get("p99_latency_ms", 0),
            "throughput":     perf_metrics.get("throughput_rps", 0),
        }
        final_json["load"] = {
            "concurrent_users":    load_metrics.get("concurrent_users", 0),
            "duration_seconds":    load_metrics.get("duration_seconds", 0),
            "total_requests":      load_metrics.get("total_requests", 0),
            "successful":          load_metrics.get("successful", 0),
            "errors":              load_metrics.get("errors", 0),
            "error_rate":          load_metrics.get("error_rate", 0),
            "avg_latency":         load_metrics.get("avg_latency_ms", 0),
            "p95_latency":         load_metrics.get("p95_latency_ms", 0),
            "requests_per_second": load_metrics.get("requests_per_second", 0),
            "tool_used":           load_metrics.get("tool_used", "locust"),
        }

        # Fix persona_breakdown pass_rate from 0-1 to 0-100
        for pb in final_json.get("persona_breakdown", []):
            if isinstance(pb, dict) and "pass_rate" in pb:
                pb["pass_rate"] = round(pb["pass_rate"] * 100, 1)

        # Fix 34: agent_name is already on the report model — no post-hoc addition needed

        # Add legacy fields for UI backward compatibility
        final_json["personas"] = [
            {
                "id": p.persona_id, "name": p.name,
                "description": p.description or p.background[:200],
                "traits": p.traits,
                "sample_prompts": p.sample_prompts or p.entry_points,
                "fishbone": p.fishbone_dimensions,
            }
            for p in personas
        ]
        final_json["quality_criteria"] = quality_criteria

        # Attach RCA to final JSON
        if report.rca:
            final_json["rca"] = report.rca.model_dump()

        job_store[run_id]["status"]   = "completed"
        job_store[run_id]["progress"] = 100
        job_store[run_id]["message"]  = f"Completed! Health score: {report.health_score:.0%}"
        job_store[run_id]["results"]  = final_json
        _job_locks.pop(run_id, None)   # release lock for completed run

        # Fix 28: persist completed run to SQLite for history / regression endpoints
        try:
            from core import db as _db
            _db.save_run(
                run_id=run_id,
                project_id=project_id,
                health_score=report.health_score,
                domain=config.agent_domain,
                application_type=config.application_type.value,
                results=final_json,
            )
        except Exception as db_err:
            print(f"[Pipeline] DB save failed (non-fatal): {db_err}")

    except Exception as e:
        import traceback
        job_store[run_id]["status"] = "failed"
        job_store[run_id]["error"]  = str(e)
        job_store[run_id]["message"] = f"Pipeline failed: {str(e)[:200]}"
        print(f"[Pipeline ERROR] {run_id}: {traceback.format_exc()}")
        _job_locks.pop(run_id, None)   # release lock on failure too
