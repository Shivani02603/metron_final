# METRON — Technical Architecture & Implementation Document

---

## 1. What METRON Is and Why It Exists

METRON is an AI application testing platform. It answers one question: **is your AI agent good enough to deploy?** Not just "does it respond" — but does it respond correctly, safely, consistently, and fast enough, across the full range of people who will actually use it?

Most AI testing today is manual. A developer writes a few test prompts, checks the output by eye, and ships. METRON replaces that with a systematic, automated pipeline that generates realistic users, runs real conversations, and evaluates the results using purpose-built tools — Presidio for PII, Detoxify for toxicity, LLM Guard for prompt injection, DeepEval for quality, RAGAS for RAG faithfulness, and Locust for load testing.

The result is a health score and a full breakdown of where the AI passed and where it failed.

---

## 2. Technology Stack

**Backend**
- Python 3.11, FastAPI, Uvicorn (ASGI)
- Pydantic v2 for all data models
- `litellm` for unified LLM routing across 4 providers
- `aiohttp` for async HTTP calls to the target AI endpoint
- `asyncio` throughout — the entire pipeline is async

**Frontend**
- Next.js (React 19, TypeScript)
- Tailwind CSS
- `sessionStorage` for project and config state
- Polling-based live feed (2500ms interval against `/api/job/{run_id}/status`)

**Evaluation Tools**
- `deepeval` 3.8.8 — HallucinationMetric, AnswerRelevancyMetric, GEval, BiasMetric, ContextualRelevancyMetric
- `ragas` 0.4.3 — faithfulness, context_recall, context_precision (batch LLM evaluation)
- `presidio-analyzer` — PII entity detection (SSN, email, phone, credit card)
- `detoxify` — BERT-based toxicity classifier (input and output)
- `llm-guard` — DeBERTa-based prompt injection scanner
- `locust` — load testing via headless subprocess

**LLM Providers**
- NVIDIA NIM (Llama 3.1 8B/70B) — default, 40 RPM free
- Azure OpenAI (GPT-4o) — 600 RPM, 100K TPM
- Groq (Llama 3.1/3.3) — 30 RPM, 100K tokens/day free
- Google Gemini (2.0 Flash / 1.5 Pro) — 60 RPM

---

## 3. System Architecture

The system is split cleanly into two services:

```
┌──────────────────────────────────────────────────────────────────┐
│                        METRON FRONTEND                           │
│                     Next.js  :3000                               │
│                                                                  │
│  /dashboard           Project hub (sessionStorage)               │
│  /configure           Test configuration form                    │
│  /preview             Test plan preview + tool status            │
│  /run                 Live execution feed (polling)              │
│  /results             Full results dashboard                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │  HTTP REST (localhost:8000)
┌────────────────────────────▼─────────────────────────────────────┐
│                        METRON BACKEND                            │
│                    FastAPI  :8000                                 │
│                                                                  │
│  POST /api/run        → starts background pipeline               │
│  GET  /api/job/{id}/status  → progress + log events             │
│  GET  /api/job/{id}/results → final JSON report                  │
│  POST /api/preview    → personas + scenarios preview             │
│  POST /api/connect-test → endpoint connectivity check            │
│  POST /api/parse-document → doc → AppProfile                     │
│  GET  /api/providers  → LLM provider list                        │
│  GET  /api/tools/status → tool availability                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  In-memory job store: Dict[run_id, Dict]                │    │
│  │  Background tasks: asyncio via FastAPI BackgroundTasks  │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                             │  async HTTP (aiohttp)
                             ▼
                   ┌─────────────────────┐
                   │  TARGET AI ENDPOINT  │
                   │  (the app under test)│
                   └─────────────────────┘
```

The pipeline runs as a FastAPI background task. The frontend polls for status every 2500ms and renders events as they arrive. The target AI endpoint receives real HTTP requests — METRON never mocks the target, it always talks to the real thing.

---

## 4. The 8-Stage Pipeline

Every test run goes through exactly 8 stages in sequence. Stages 4a/4b/4c run in parallel.

```
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4a ─┐
                                                     ├→ Stage 5 → Stage 6 → Stage 7
                                          Stage 4b ─┤
                                          Stage 4c ─┘
                                          Stage 4d
                                          Stage 4e
                                          Stage 4f (RAG only)
```

### Stage 0 — App Profile Extraction (`s0_profile/document_parser.py`)

The user uploads a document describing their AI application. This stage extracts structured metadata via an LLM call:

```python
AppProfile {
    application_type: chatbot | rag | multi_agent | form
    domain: "healthcare" | "finance" | "legal" | ...
    user_types: ["patient", "nurse", "administrator"]
    use_cases: ["check symptoms", "find nearby clinics"]
    boundaries: ["does not prescribe medication"]
    domain_vocabulary: ["triage", "dosage", "comorbidity"]
    success_criteria: {"response_time": "<2s", "accuracy": ">90%"}
}
```

If no document is provided, `build_profile_from_config()` constructs a minimal profile from the form fields (agent description + domain + application type).

### Stage 1 — Persona Generation (`s1_personas/`)

**1a — Fishbone Coverage Builder (`fishbone_builder.py`)**

The fishbone matrix ensures systematic persona coverage across 5 dimensions:

```
Dimension         Values
─────────────     ──────────────────────────────────
user_type         from AppProfile.user_types
expertise         novice | intermediate | expert
emotional_state   calm | frustrated | urgent
intent            genuine | adversarial | edge_case
goal_type         from AppProfile.use_cases
```

Domain matters here. A `healthcare` domain triggers more adversarial and edge case slots (4 adversarial, 3 edge case). A `retail` domain triggers more emotional/frustrated genuine users (4 emotional). This is hardcoded in `HIGH_SECURITY_DOMAINS` and `HIGH_TRAFFIC_DOMAINS` in `config.py`.

Each combination is a "slot" — a description of who should be generated, not the persona itself.

**1b — Persona Builder (`persona_builder.py`)**

For each slot, one LLM call generates a full persona JSON:

```python
Persona {
    name, background, goal
    mental_model: { believes_can[], doesnt_know[] }
    language_model: { base_style, frustrated_style, vocabulary_prefer[], vocabulary_avoid[] }
    behavioral_params: { patience_level(1-5), persistence(1-10),
                         escalation_trigger, abandon_trigger }
    entry_points: ["first thing they'd say to this AI"]
    traits: ["detail-oriented", "technically anxious"]
    adversarial_goal: "extract system prompt" (adversarial personas only)
    attack_category: "prompt_injection" (adversarial personas only)
    fishbone_dimensions: { expertise, emotional_state, intent, ... }
}
```

All personas are generated in parallel (`asyncio.gather`) with a semaphore of 5.

**1c — Coverage Validator (`coverage_validator.py`)**

After generation, a validator checks for gaps (e.g., no frustrated user, no expert). If gaps found, up to 3 extra slots are added and personas generated to fill them.

### Stage 2 — Test Generation

**2a — Functional Prompts (`functional_gen.py`)**

For each persona, 3 test prompts are generated in that persona's exact voice:
- Prompt 1: how they'd first approach the app
- Prompt 2: an alternative angle on the same goal
- Prompt 3: a follow-up after a vague/partial response

Each prompt has an `expected_behavior` field (what a correct response should contain) and an `expected_answer` field for RAG mode.

For RAG mode with a ground truth file, this stage skips LLM generation entirely and converts the uploaded Q&A pairs directly into `GeneratedPrompt` objects via `_ground_truth_to_prompts()`. The context field from the ground truth file becomes `ground_truth_context` — the only source of context used in evaluation.

**2b — Security Prompts (`security_gen.py`)**

Only runs for adversarial personas. Loads 25 hand-crafted attack templates from `attacks.json` (organized by category: jailbreak, prompt_injection, pii_extraction, toxicity, social_engineering, encoding). For each selected category, up to N attacks per category are selected and adapted to the application's domain via an LLM call that disguises the attack as a plausible user message.

Golden dataset prompts (AdvBench for injection, HarmBench for toxic requests) are downloaded from HuggingFace and distributed round-robin across adversarial personas.

**2c — Quality Criteria (`quality_criteria.py`)**

One LLM call generates domain-specific evaluation criteria for GEval (e.g., for healthcare: "Clinical Accuracy", "Empathy", "Safety Compliance").

### Stage 3 — Conversation Execution (`s3_execution/conversation_runner.py`)

This is where METRON actually talks to the target AI. For each persona + prompt pair, it runs a multi-turn conversation.

**Turn 1 — Entry Point**
The generated prompt is sent directly to the target AI endpoint via the appropriate adapter (ChatbotAdapter, RAGAdapter, MultiAgentAdapter, FormAdapter). The adapter sends an HTTP POST with `{request_field: message}` and extracts the response from `response_field` (dot-notation supported: `output.text.value`).

**Turns 2–3 — Combined Eval+Generate**
A single LLM call does two things at once: evaluates the AI's response and generates the persona's next message. This is the `COMBINED_TURN_PROMPT`. It returns:

```json
{
  "response_type": "helpful | vague | deflecting | wrong | honest_limitation | harmful",
  "goal_progress": "achieved | partial | none",
  "new_state": "seeking | clarifying | frustrated | escalating | satisfied | abandoned",
  "reasoning": "...",
  "next_message": "..." or null
}
```

The conversation ends early if `goal_progress == "achieved"`, `next_message == null`, or `new_state` is `satisfied` or `abandoned`.

**For RAG mode**, the conversation always stops after turn 1. The `ground_truth_context` from the prompt is used as `retrieved_context` — not whatever the endpoint returns. This is the only source of context for RAGAS evaluation.

**Rate limiting:** A semaphore of 3 limits concurrent conversations. A 429 from the target endpoint triggers a 2-second sleep and one retry.

All conversations run in parallel via `asyncio.gather`.

### Stage 4 — Evaluation (all run in parallel)

**4a — Functional (`s4_evaluation/functional.py`)**

For each functional conversation's last turn:

| Metric | Tool | Notes |
|--------|------|-------|
| hallucination | DeepEval HallucinationMetric | score = 1.0 - raw (inverted) |
| answer_relevancy | DeepEval AnswerRelevancyMetric | direct score |
| usefulness | DeepEval GEval | custom usefulness criteria |
| llm_relevance / llm_accuracy | LLM Judge | domain-specific criteria |

All DeepEval calls run in a `ThreadPoolExecutor` via `loop.run_in_executor` (DeepEval is synchronous). If any metric throws, that metric is skipped for that conversation — no fabricated score is recorded.

**4b — Security (`s4_evaluation/security.py`)**

Runs on ALL conversations for PII/toxicity/bias, only on security conversations for injection/toxic_request/attack_resistance.

| Metric | Tool | Scope |
|--------|------|-------|
| pii_leakage | Presidio AnalyzerEngine | all convs, AI response |
| toxicity | Detoxify (BERT) | all convs, AI response |
| bias_fairness | DeepEval BiasMetric | non-security convs only |
| prompt_injection | LLM Guard (DeBERTa) | security convs with attack_category=prompt_injection |
| toxic_request | heuristic refusal check | security convs with attack_category=toxic_request |
| attack_resistance | LLM Judge rubric | all security convs |

Noisy Presidio entity types are filtered out: `DATE_TIME`, `NRP`, `LOCATION`, `URL`, `PERSON` — these produce false positives in normal AI responses.

For HTTP errors: 502 = gateway blocked attack = defense succeeded (pass). 500 = AI crashed on attack input = vulnerability (fail).

**4c — Quality (`s4_evaluation/quality.py`)**

Skips security conversations entirely (attack prompts trigger Azure content filter on GEval). For functional conversations:
- GEval with domain-specific criteria (up to 6 criteria, each scored 0–1)
- RAGAS (RAG mode only): faithfulness, answer_relevancy, context_recall, context_precision

**4d — Performance (`s4_evaluation/performance.py`)**

Sends N requests (default 20, cycling through 5 test prompts) in batches of 5 with 0.2s sleep between batches. Measures: avg, min, max, median, p50, p95, p99 latency and throughput (RPS). Pass threshold: p95 ≤ 5000ms and error rate < 5%.

**4e — Load (`s4_evaluation/load.py`)**

Generates a Locust file at runtime with the endpoint config and test prompts, then runs it as a subprocess:
```bash
python -m locust --headless -u {users} -r {spawn_rate} -t {duration}s --csv={csv_path}
```
Spawn rate = `max(1, users // 5)` (gradual ramp-up). Timeout = `duration + max(60, users * 3)`. Parses CSV output for RPS, error rate, p95 latency.

**4f — RAG (`s4_evaluation/rag.py`)**

Only runs when `config.is_rag = True` and conversations have `retrieved_context`.

RAGAS runs as a batch (one API call per metric, covering all conversations together):
```python
EvaluationDataset(samples=[
    SingleTurnSample(user_input, response, retrieved_contexts, reference)
    ...
])
ragas_evaluate(dataset, metrics=[faithfulness, context_recall, context_precision])
```

DeepEval runs per-conversation (semaphore of 3): AnswerRelevancyMetric + ContextualRelevancyMetric.

Both tools route through Azure GPT-4o via `LangchainLLMWrapper(AzureChatOpenAI(...))` for RAGAS and `DeepEvalBaseLLM` subclass for DeepEval.

### Stage 5 — Aggregation (`s5_aggregation/aggregator.py`)

Produces the `AggregatedReport`. Key computation: **domain-weighted health score**.

```python
health_score = Σ (class_avg_score × domain_weight[class]) / Σ domain_weight[class]
```

Domain weights from `config.py` DOMAIN_WEIGHTS:
- Finance/Banking: security 40%, functional 35%, quality 10%, performance 10%, load 5%
- Medical/Healthcare: security 45%, functional 35%, quality 10%
- Travel/Ecommerce: performance 25%, load 15%, functional 35%, security 15%
- Default: functional 40%, security 30%, quality 10%, performance 15%, load 5%

Pass threshold: `health_score ≥ 0.70`.

Also builds: per-class summaries, per-persona breakdown (sorted worst-first), top-20 failure drill-down with full conversation turns.

### Stage 6 — Feedback Loop (`s6_feedback/feedback_loop.py`)

Runs one iteration of adaptive persona generation. Analyzes which personas found failures (failure rate > 20% = effective). If effective personas exist, the LLM analyzes the top 15 failures and recommends up to 3 new persona slots targeting those failure patterns. Stages 1b, 2, 3, 4, 5 run again on the new personas. The report is re-aggregated with 70% original + 30% feedback weighting.

If no effective personas found (all pass rates are similar), the loop exits early.

### Stage 7 — Report Generation (`s7_report/report_generator.py`)

`report_to_json()` serializes the AggregatedReport to a frontend-ready dict. `generate_html_report()` produces a styled HTML document with health score gauge, per-class breakdown tables, persona breakdown, and failure drill-down.

The pipeline normalizes all field names and scales (pass_rate 0–1 → 0–100%) before storing in the job store.

---

## 5. LLM Client Architecture (`core/llm_client.py`)

Every LLM call in the pipeline goes through one class: `LLMClient`.

**Rate Limiting**
`RateLimiter` implements a token bucket with 20% safety headroom. It reserves time slots under an asyncio lock but sleeps outside the lock — so concurrent callers can all reserve their slots and sleep simultaneously rather than queuing.

```
interval = 60.0 / (rpm × 0.8)   ← 20% headroom
next_allowed slot is reserved atomically
sleep happens concurrently outside the lock
```

**Task Routing**
Three task types map to different model tiers:
- `fast` → lightweight model (Llama 3.1-8B, Gemini Flash) — used for persona generation, test generation
- `judge` → heavyweight model (Llama 3.3-70B, GPT-4o) — used for evaluation rubrics
- `balanced` → middle tier — default

**Fallback Chain**
On 429 or quota exhaustion, the client automatically falls back to: `groq/llama-3.3-70b-versatile` → `groq/llama-3.1-8b-instant`. The primary provider is retried on the next call.

**JSON Extraction**
`complete_json()` retries up to 4 times with decreasing temperature. Extraction attempts: direct `json.loads` → strip markdown fences → greedy brace/bracket extraction.

---

## 6. Adapter Layer (`core/adapters/`)

Four adapters handle the actual HTTP calls to the target AI:

**ChatbotAdapter** — generic REST/JSON. Sends `POST {request_field: message}`, extracts response from `response_field` using dot-notation (e.g., `data.output.text` → `resp["data"]["output"]["text"]`). Measures wall-clock latency in milliseconds.

**RAGAdapter** — same as chatbot but also extracts `retrieved_context` (list of strings) and `source_documents` from the response.

**MultiAgentAdapter** — extracts `agent_trace` (list of `{agent, action, output}` dicts) representing the multi-step reasoning chain.

**FormAdapter** — submits form-style payloads and handles redirect responses.

All adapters return `AdapterResponse(ok, text, latency_ms, error, retrieved_context, agent_trace)`.

---

## 7. Data Models (`core/models.py`)

The full data flow through the pipeline:

```
RunConfig
    ↓ Stage 0
AppProfile
    ↓ Stage 1
List[Persona]
    ↓ Stage 2
List[GeneratedPrompt]
    ↓ Stage 3
List[Conversation]  (each has List[ConversationTurn])
    ↓ Stage 4
List[MetricResult]  (flat — one row per metric per conversation)
    ↓ Stage 5
AggregatedReport    (health_score, test_classes, persona_breakdown, failure_drill_down)
    ↓ Stage 7
Dict (JSON for frontend)
```

`MetricResult` is the core evaluation atom:
```python
MetricResult {
    conversation_id, persona_id, persona_name, intent, fishbone
    prompt, response, latency_ms, superset, metric_name
    score (0.0–1.0), passed (bool), reason (str)
    vulnerability_found, owasp_category, severity   # security extras
    pii_detected, pii_types                         # PII extras
}
```

---

## 8. Known Limitations and Design Decisions

**In-memory job store** — the `jobs: Dict` in `fastapi_server.py` is process-local. Jobs are lost on server restart. This was intentional for the MVP to keep the stack simple (no database dependency). The trade-off is no persistence and no multi-process scaling.

**sessionStorage coupling** — the frontend stores all project configuration in `sessionStorage` keyed by `projectId`. This was chosen to avoid a user authentication system. The trade-off is data is lost on page refresh or when switching browsers.

**Security conversations skip quality GEval** — attack prompts (jailbreak text, harmful requests) trigger Azure OpenAI's content filter when passed as GEval evaluation input. The fix is to skip quality metrics for all `TestClass.SECURITY` conversations. This is handled in both `quality.py` and `security.py`.

**RAG context is ground truth only** — when a ground truth file is uploaded, `ground_truth_context` becomes the only source of retrieved_context. The endpoint's own returned context is ignored. This is deliberate: the user-provided ground truth is the authoritative reference, not what the RAG system happens to retrieve.

**BiasMetric skips security conversations** — same reason as above. Attack prompts trigger content filter on BiasMetric evaluation. The check is `if conv.test_class != TestClass.SECURITY` in `security.py`.

**RAGAS 0.4.x API** — the installed version (0.4.3) uses `EvaluationDataset` + `SingleTurnSample` instead of HuggingFace `Dataset`. Both `rag.py` and `quality.py` try the 0.4.x API first and fall back to the 0.1.x `Dataset.from_dict()` approach.

---

## 9. Configuration Reference

**Environment Variables Required**

| Variable | Used By | Required For |
|----------|---------|--------------|
| `AZURE_OPENAI_API_KEY` | LLMClient, RAGAS, DeepEval | Azure provider |
| `AZURE_OPENAI_ENDPOINT` | DeepEval, RAGAS | Azure provider |
| `AZURE_API_VERSION` | LLMClient | Azure provider (default: 2025-01-01-preview) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | RAGAS | RAGAS Azure config (default: gpt-4o) |
| `GROQ_API_KEY` | LLMClient | Groq provider |
| `GEMINI_API_KEY` | LLMClient | Gemini provider |
| `NVIDIA_NIM_API_KEY` | LLMClient | NVIDIA NIM |
| `OPENAI_API_VERSION` | LangchainLLMWrapper | RAGAS (set by `_set_azure_env`) |

**Starting the Backend**
```bash
cd metron-unified
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # for Presidio PII detection
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

**Starting the Frontend**
```bash
cd metron-ai
npm install
npm run dev   # runs on :3000
```

---

## 10. End-to-End Request Lifecycle (Full Run)

```
1. User submits form on /configure
   └→ sessionStorage["config_{id}"] = RunConfig JSON

2. /preview page calls POST /api/run with FormData
   ├→ config: JSON string
   ├→ document: uploaded file (optional)
   └→ ground_truth_file: CSV/JSON (RAG mode)

3. FastAPI assigns run_id, starts background task, returns {run_id}

4. Background task: run_pipeline(run_id, config, job_store)
   Stage 0 → parse document → AppProfile
   Stage 1 → fishbone slots → LLM personas → coverage check
   Stage 2 → functional prompts + security prompts + quality criteria
   Stage 3 → run all conversations (semaphore=3, retry on 429)
   Stage 4a/4b/4c → asyncio.gather(functional, security, quality)
   Stage 4f → RAG evaluation (if is_rag)
   Stage 4d → performance (N async requests)
   Stage 4e → load (Locust subprocess)
   Stage 5 → aggregate → health score
   Stage 6 → feedback loop (if enabled)
   Stage 7 → generate HTML report
   └→ job_store[run_id]["results"] = final_json

5. Frontend polls GET /api/job/{run_id}/status every 2500ms
   └→ renders log_events[] as live feed cards

6. On status == "completed":
   └→ router.push(/results) → fetches GET /api/job/{run_id}/results
   └→ renders full results dashboard
```

Total wall-clock time for a typical run (3 personas, default settings, Groq provider):
- Stages 0–2: ~30–60 seconds (LLM generation)
- Stage 3: ~60–120 seconds (conversations at target endpoint)
- Stage 4: ~60–180 seconds (DeepEval + RAGAS calls)
- Stage 5–7: ~10 seconds
- **Total: 3–6 minutes** (varies significantly by provider RPM and endpoint latency)
