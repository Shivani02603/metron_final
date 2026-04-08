"""
Master 8-stage pipeline orchestrator.
Connects all stages, updates job_store progress at each stage,
and handles the full lifecycle from AppProfile → AggregatedReport.
"""

from __future__ import annotations
import asyncio
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.llm_client import LLMClient
from core.models import (
    AppProfile, AggregatedReport, Conversation, GeneratedPrompt,
    MetricResult, Persona, RunConfig,
)

# Stage imports
from stages.s0_profile.document_parser import parse_document, build_profile_from_config
from stages.s1_personas.fishbone_builder import build_slots
from stages.s1_personas.persona_builder import build_all_personas, build_persona
from stages.s1_personas.coverage_validator import validate_and_fill
from stages.s2_tests.functional_gen import generate_all_functional, generate_performance_prompts
from stages.s2_tests.security_gen import generate_all_security
from stages.s2_tests.quality_criteria import generate_quality_criteria
from stages.s3_execution.conversation_runner import run_all_conversations
from stages.s4_evaluation.functional import evaluate_functional
from stages.s4_evaluation.security import evaluate_security
from stages.s4_evaluation.quality import evaluate_quality
from stages.s4_evaluation.performance import evaluate_performance
from stages.s4_evaluation.load import evaluate_load
from stages.s5_aggregation.aggregator import aggregate
from stages.s6_feedback.feedback_loop import run_feedback_loop
from stages.s7_report.report_generator import report_to_json, generate_html_report


# ── Progress helpers ───────────────────────────────────────────────────────

def _update(job_store: Dict, run_id: str, progress: int, message: str, phase: str = "", phase_data: Dict = {}):
    if run_id in job_store:
        job_store[run_id]["progress"]      = progress
        job_store[run_id]["message"]       = message
        job_store[run_id]["current_phase"] = phase
        if phase_data:
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

        # ── Stage 1: Persona Generation (Fishbone) ─────────────────────────
        _update(job_store, run_id, 10, "Building persona coverage matrix…", "personas")
        slots = build_slots(profile, num_personas=config.num_personas)
        personas = await build_all_personas(slots, profile, llm_client, project_id)

        # Coverage validation (adds up to 3 more slots if gaps found)
        extra_slots = await validate_and_fill(personas, profile, llm_client)
        if extra_slots:
            extra_personas = await build_all_personas(extra_slots, profile, llm_client, project_id)
            personas.extend(extra_personas)

        _update(job_store, run_id, 18, f"Generated {len(personas)} personas", "personas",
                {"count": len(personas), "names": [p.name for p in personas]})

        # ── Stage 2: Domain-Specific Test Generation ───────────────────────
        _update(job_store, run_id, 20, "Generating domain-specific test prompts…", "test_gen")

        # 2a: Functional prompts
        func_prompts = await generate_all_functional(personas, profile, llm_client)

        # 2b: Security prompts (adversarial personas only)
        sec_prompts = await generate_all_security(
            personas, profile, llm_client,
            selected_categories=config.selected_attacks,
            attacks_per_category=config.attacks_per_category,
        )

        # 2c: Quality criteria
        quality_criteria = await generate_quality_criteria(profile, llm_client)

        all_prompts = func_prompts + sec_prompts
        _update(job_store, run_id, 25, f"Generated {len(all_prompts)} test prompts", "test_gen",
                {"functional": len(func_prompts), "security": len(sec_prompts)})

        # ── Stage 3+4: Execution + Evaluation (interleaved) ────────────────
        _update(job_store, run_id, 28, "Running conversations with target AI…", "execution")

        # Progress callback for live updates
        completed_convs: List[Conversation] = []
        def on_conv(done: int, total: int, conv: Conversation):
            progress = 28 + int((done / total) * 30)   # 28-58%
            phase = conv.test_class.value
            _update(job_store, run_id, progress,
                    f"Executing {phase} test {done}/{total}…", "execution")
            completed_convs.append(conv)

        conversations = await run_all_conversations(
            personas, all_prompts, config, llm_client, project_id,
            progress_callback=on_conv,
        )

        # Stage 4: Functional evaluation
        _update(job_store, run_id, 58, "Evaluating functional test results…", "functional")
        func_results = await evaluate_functional(conversations, personas, config, llm_client, quality_criteria)

        # Stage 4: Security evaluation
        _update(job_store, run_id, 65, "Evaluating security test results…", "security")
        sec_results = await evaluate_security(conversations, personas, config, llm_client)

        # Stage 4: Quality evaluation
        _update(job_store, run_id, 70, "Evaluating quality metrics…", "quality")
        qual_results = await evaluate_quality(conversations, personas, config, llm_client, quality_criteria)

        all_metric_results = func_results + sec_results + qual_results

        # Update phase results for live UI polling
        _update(job_store, run_id, 72, "Functional + Security + Quality evaluated",
                "evaluation", {
                    "functional": {"count": len(func_results), "passed": sum(1 for r in func_results if r.passed)},
                    "security":   {"count": len(sec_results),  "passed": sum(1 for r in sec_results  if r.passed)},
                    "quality":    {"count": len(qual_results),  "passed": sum(1 for r in qual_results if r.passed)},
                })

        # Stage 4: Performance evaluation
        _update(job_store, run_id, 75, "Running performance tests…", "performance")
        perf_metrics = await evaluate_performance(config, run_id=run_id)
        _update(job_store, run_id, 82, f"Performance: p95={perf_metrics.get('p95_latency_ms', 0):.0f}ms",
                "performance", perf_metrics)

        # Stage 4: Load evaluation
        _update(job_store, run_id, 84, f"Running load test ({config.load_concurrent_users} concurrent users)…", "load")
        load_metrics = await evaluate_load(config)
        _update(job_store, run_id, 89, f"Load: {load_metrics.get('requests_per_second', 0):.1f} RPS",
                "load", load_metrics)

        # ── Stage 5: Aggregation ───────────────────────────────────────────
        _update(job_store, run_id, 90, "Aggregating results…", "aggregation")
        report = aggregate(
            metric_results=all_metric_results,
            conversations=conversations,
            personas=personas,
            config=config,
            performance_metrics=perf_metrics,
            load_metrics=load_metrics,
            run_id=run_id,
            project_id=project_id,
        )

        # ── Stage 6: Feedback Loop ─────────────────────────────────────────
        if config.enable_feedback_loop:
            _update(job_store, run_id, 90, "Running adaptive feedback loop…", "feedback")

            async def _run_stages_on_new_slots(
                new_slots: List[Dict], profile: AppProfile,
                config: RunConfig, llm_client: LLMClient,
            ) -> Tuple[List[MetricResult], List[Conversation], List[Persona]]:
                new_personas = await build_all_personas(new_slots, profile, llm_client, project_id)
                new_func = await generate_all_functional(new_personas, profile, llm_client)
                new_sec  = await generate_all_security(new_personas, profile, llm_client,
                                                       config.selected_attacks, config.attacks_per_category)
                new_convs = await run_all_conversations(new_personas, new_func + new_sec, config, llm_client, project_id)
                new_func_r = await evaluate_functional(new_convs, new_personas, config, llm_client, quality_criteria)
                new_sec_r  = await evaluate_security(new_convs, new_personas, config, llm_client)
                new_qual_r = await evaluate_quality(new_convs, new_personas, config, llm_client, quality_criteria)
                return new_func_r + new_sec_r + new_qual_r, new_convs, new_personas

            def fb_progress(pct: int, msg: str):
                _update(job_store, run_id, pct, msg, "feedback")

            original_persona_count = len(personas)
            report, personas = await run_feedback_loop(
                report=report,
                profile=profile,
                personas=personas,
                config=config,
                llm_client=llm_client,
                run_stages_fn=_run_stages_on_new_slots,
                progress_callback=fb_progress,
            )
            new_personas_added = len(personas) - original_persona_count
            _update(job_store, run_id, 97, "Feedback loop complete", "feedback", {
                "new_personas_count": new_personas_added,
                "effective_personas": sum(
                    1 for pb in report.persona_breakdown
                    if hasattr(pb, "pass_rate") and pb.pass_rate < 50
                ),
            })

        # ── Stage 7: Report Generation ─────────────────────────────────────
        _update(job_store, run_id, 97, "Generating report…", "report")
        report.report_html = generate_html_report(report)
        final_json = report_to_json(report)
        _update(job_store, run_id, 99, "Report ready", "report", {"generated": True})

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

        job_store[run_id]["status"]   = "completed"
        job_store[run_id]["progress"] = 100
        job_store[run_id]["message"]  = f"Completed! Health score: {report.health_score:.0%}"
        job_store[run_id]["results"]  = final_json

    except Exception as e:
        import traceback
        job_store[run_id]["status"] = "failed"
        job_store[run_id]["error"]  = str(e)
        job_store[run_id]["message"] = f"Pipeline failed: {str(e)[:200]}"
        print(f"[Pipeline ERROR] {run_id}: {traceback.format_exc()}")
