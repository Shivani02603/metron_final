"""
Unified Pydantic schemas for the entire pipeline.
Sourced primarily from new metron-backend/app/core/models.py,
extended with fields from existing METRON TestConfig/TestResult dataclasses.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


# ── Enumerations ───────────────────────────────────────────────────────────

class ApplicationType(str, Enum):
    CHATBOT     = "chatbot"
    RAG         = "rag"
    MULTI_AGENT = "multi_agent"
    FORM        = "form"

class TestClass(str, Enum):
    FUNCTIONAL  = "functional"
    SECURITY    = "security"
    QUALITY     = "quality"
    PERFORMANCE = "performance"
    LOAD        = "load"

class PersonaIntent(str, Enum):
    GENUINE     = "genuine"
    ADVERSARIAL = "adversarial"
    EDGE_CASE   = "edge_case"

class ExpertiseLevel(str, Enum):
    NOVICE       = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT       = "expert"

class EmotionalState(str, Enum):
    CALM       = "calm"
    FRUSTRATED = "frustrated"
    URGENT     = "urgent"

class ConversationState(str, Enum):
    SEEKING    = "seeking"
    CLARIFYING = "clarifying"
    FRUSTRATED = "frustrated"
    ESCALATING = "escalating"
    SATISFIED  = "satisfied"
    ABANDONED  = "abandoned"

class ResponseType(str, Enum):
    HELPFUL           = "helpful"
    VAGUE             = "vague"
    DEFLECTING        = "deflecting"
    WRONG             = "wrong"
    HONEST_LIMITATION = "honest_limitation"
    HARMFUL           = "harmful"


# ── Stage 0: App Profile ───────────────────────────────────────────────────

class AgentDefinition(BaseModel):
    name:         str
    role:         str
    capabilities: List[str] = []

class AppProfile(BaseModel):
    project_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    application_type: ApplicationType = ApplicationType.CHATBOT
    agents:           List[AgentDefinition] = []
    user_types:       List[str] = []
    use_cases:        List[str] = []
    domain:           str = "general"
    domain_vocabulary: List[str] = []
    boundaries:       List[str] = []
    success_criteria: Dict[str, str] = {}
    raw_document:     str = ""


# ── Run Configuration (from configure form) ────────────────────────────────

class RunConfig(BaseModel):
    """Full configuration submitted by user from the configure page."""
    # Project identity (separate from run_id so multiple runs share a project)
    project_id:      Optional[str] = None   # UI sends dashboard project [id]; pipeline generates one if absent

    # Endpoint
    endpoint_url:    str
    request_field:   str = "message"
    response_field:  str = "response"
    auth_type:       str = "none"    # "none" or "bearer"
    auth_token:      str = ""

    # Agent / App profile
    agent_name:        str = ""
    agent_description: str = ""
    agent_domain:      str = "general"
    application_type:  ApplicationType = ApplicationType.CHATBOT

    # RAG
    is_rag:       bool = False
    rag_text:     str  = ""
    ground_truth: List[Dict] = []   # [{"question": ..., "expected_answer": ..., "context": str|list}]

    # Test parameters
    num_personas:           int  = 3
    num_scenarios:          int  = 5
    conversation_turns:     int  = 3
    enable_judge:           bool = True
    performance_requests:   int  = 20
    load_concurrent_users:  int  = 5
    load_duration_seconds:  int  = 30

    # LLM
    llm_provider: str = "Groq"
    llm_api_key:  str = ""

    # Security
    selected_attacks:      List[str] = ["jailbreak", "prompt_injection", "pii_extraction", "toxicity", "encoding"]
    attacks_per_category:  int = 3

    # Quality
    ragas_metrics:   List[str] = ["faithfulness", "answer_relevancy"]
    deepeval_metrics: List[str] = ["hallucination", "toxicity"]
    use_geval:       bool = True

    # Feedback loop
    enable_feedback_loop: bool = True

    # ── Advanced request/response templating ──────────────────────────────
    # Set request_template to a full JSON body string with placeholders:
    #   {{query}}           — replaced with the test message
    #   {{uuid}}            — replaced with a new UUID on every request
    #   {{conversation_id}} — replaced with a UUID stable for all turns of one conversation
    # When set, request_field is ignored for body construction (response_field still used for extraction).
    request_template:     Optional[str] = None

    # Trim response text at this marker (e.g. "FOLLOW UP QUESTIONS").
    # Everything at and after the marker is discarded before evaluation.
    response_trim_marker: Optional[str] = None

    # ── Architecture profile for RCA (Stage 8) ────────────────────────────
    # Core infrastructure
    deployment_type:   str = "unknown"   # "serverless" | "server" | "container" | "unknown"
    vector_db:         str = ""          # "pinecone" | "weaviate" | "qdrant" | "faiss" | "chroma" | ""
    session_db:        str = ""          # "redis" | "postgresql" | "mongodb" | "dynamodb" | "sqlite" | ""
    # Extended infrastructure
    cache_layer:       str = ""          # "redis" | "memcached" | "cdn" | "in_memory" | ""
    message_queue:     str = ""          # "kafka" | "rabbitmq" | "sqs" | "pubsub" | "kinesis" | ""
    api_gateway:       str = ""          # "aws_apigw" | "azure_apim" | "kong" | "nginx" | ""
    auth_mechanism:    str = ""          # "oauth" | "api_key" | "jwt" | "saml" | ""
    monitoring_tool:   str = ""          # "datadog" | "cloudwatch" | "prometheus" | "grafana" | "newrelic" | ""
    is_multi_region:   bool = False
    # Resilience flags
    has_rate_limiting: bool = False
    rate_limit_rpm:    Optional[int] = None
    has_retry_logic:   bool = False
    has_circuit_breaker: bool = False
    has_caching:       bool = False
    has_dlq:           bool = False      # dead-letter queue
    # Free-form / uploaded content
    additional_architecture_notes: str = ""   # free-form text (parsed by RCA mapper)
    architecture_document:         str = ""   # full text from uploaded doc / LLM-extracted diagram desc


# ── Stage 1: Personas ──────────────────────────────────────────────────────

class BehavioralParameters(BaseModel):
    patience_level:     int = 3      # 1-5
    persistence:        int = 3      # 1-10
    rephrase_strategy:  str = "simpler_words"
    escalation_trigger: str = "After 2 unhelpful responses"
    abandon_trigger:    str = "After 4 failed attempts"

class LanguageModel(BaseModel):
    base_style:       str = ""
    frustrated_style: str = ""
    vocabulary_avoid: List[str] = []
    vocabulary_prefer: List[str] = []

class Persona(BaseModel):
    persona_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id:       str = ""
    name:             str
    user_type:        str = ""
    expertise:        ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    emotional_state:  EmotionalState = EmotionalState.CALM
    intent:           PersonaIntent  = PersonaIntent.GENUINE
    description:      str = ""       # short summary (for existing UI compatibility)
    background:       str = ""
    goal:             str = ""
    mental_model:     Dict[str, List[str]] = {}
    language_model:   LanguageModel = Field(default_factory=LanguageModel)
    reaction_model:   str = ""
    domain_knowledge: str = ""
    adversarial_goal: Optional[str] = None
    attack_category:  Optional[str] = None
    behavioral_params: BehavioralParameters = Field(default_factory=BehavioralParameters)
    traits:           List[str] = []         # for existing UI persona cards
    sample_prompts:   List[str] = []         # for existing UI persona cards (= entry_points)
    entry_points:     List[str] = []
    initial_state:    ConversationState = ConversationState.SEEKING
    max_turns:        int = 8
    fishbone_dimensions: Dict[str, str] = {}


# ── Stage 2: Generated Test Prompts ───────────────────────────────────────

class GeneratedPrompt(BaseModel):
    prompt_id:         str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona_id:        str
    test_class:        TestClass
    text:              str
    expected_behavior:    Optional[str]       = None
    expected_answer:      Optional[str]       = None        # RAG ground truth exact answer
    ground_truth_context: Optional[List[str]] = None        # RAG ground truth context chunks
    attack_category:      Optional[str]       = None        # security only
    owasp_category:    Optional[str] = None   # e.g. "LLM01_prompt_injection"
    severity:          Optional[str] = None   # critical/high/medium
    compliance_tags:   List[str] = []
    turn_number:       int = 1


# ── Stage 3: Conversation ──────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    turn_number:        int
    query:              str
    response:           str
    latency_ms:         float
    expected_behavior:  Optional[str] = None         # ground truth from Stage 2
    expected_answer:    Optional[str] = None         # RAG ground truth exact answer
    tokens_input:       Optional[int] = None
    tokens_output:      Optional[int] = None
    retrieved_context:  Optional[List[str]] = None   # RAG only
    agent_trace:        Optional[List[Dict[str, Any]]] = None   # multi-agent only
    persona_state:      ConversationState = ConversationState.SEEKING
    response_type:      Optional[ResponseType] = None
    is_error_response:  bool = False                  # True when chatbot returned an HTTP error
    timestamp:          datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id:      str = ""
    persona_id:      str
    persona_name:    str = ""
    test_class:      TestClass
    attack_category: Optional[str] = None   # set from GeneratedPrompt (e.g. "toxic_request", "prompt_injection")
    turns:           List[ConversationTurn] = []
    final_state:     Optional[ConversationState] = None
    goal_achieved:   Optional[bool] = None
    total_latency_ms: float = 0.0
    started_at:      datetime = Field(default_factory=datetime.utcnow)
    ended_at:        Optional[datetime] = None


# ── Stage 4: Evaluation Results ────────────────────────────────────────────

class MetricResult(BaseModel):
    """Single metric evaluation result (flat dict for aggregation)."""
    conversation_id: str
    persona_id:      str
    persona_name:    str
    intent:          str = "genuine"
    fishbone:        Dict[str, str] = {}   # expertise, emotional_state, intent
    prompt:          str
    response:        str
    latency_ms:      float
    superset:        str   # functional | security | safety_passive | quality | performance | load
    metric_name:     str
    score:           float  # 0.0-1.0
    passed:          bool
    reason:          str
    # Skipped metric tracking (Issue 36)
    # When an evaluation tool fails (rate limit, exception), record skipped=True
    # instead of omitting the result. Aggregator excludes skipped from pass-rate denominator.
    skipped:         bool = False
    skip_reason:     Optional[str] = None
    # Turn tracking for multi-turn evaluation
    turn_number:     Optional[int] = None
    # Security extras
    vulnerability_found: Optional[bool] = None
    owasp_category:      Optional[str]  = None
    severity:            Optional[str]  = None
    pii_detected:        Optional[bool] = None
    pii_types:           Optional[List[str]] = None
    # Per-prompt failure classification (Stage 8 sub-module)
    failure_taxonomy_id:    Optional[str] = None   # e.g. "C1.9"
    failure_taxonomy_label: Optional[str] = None   # e.g. "Missing Few-Shot Examples in System Prompt"
    failure_reason:         Optional[str] = None   # 2-3 sentence specific explanation


# ── Stage 5: Aggregated Report ─────────────────────────────────────────────

class ClassSummary(BaseModel):
    total:        int
    passed:       int
    failed:       int
    skipped:      int = 0          # metrics skipped due to API errors / tool failures
    pass_rate:    float
    avg_score:    float
    by_metric:    Dict[str, Any] = {}
    failures:     List[Dict[str, Any]] = []
    evaluation_warnings: List[str] = []   # populated when >50% of a metric's evals were skipped

class PersonaBreakdown(BaseModel):
    persona_id:   str
    persona_name: str
    user_type:    str = ""
    intent:       str
    fishbone:     Dict[str, str] = {}
    total:        int
    passed:       int
    failed:       int
    avg_score:    float
    pass_rate:    float

class RCAFinding(BaseModel):
    """A single probable root cause with probability and evidence."""
    rank:          int
    id:            str            # e.g. "C1.1"
    label:         str            # human-readable failure point name
    category:      str            # e.g. "AI / Model"
    category_id:   str            # e.g. "C1"
    probability:   float          # 0.0 – 1.0
    affected_count: int           # number of failed prompts this likely caused
    evidence:      List[str]      # list of observed signals that triggered this
    remediation:   str            # short fix hint
    reason:        str = ""       # detailed narrative explanation of why this was flagged

class RCAReport(BaseModel):
    """Output of Stage 8: Root Cause Analysis."""
    total_failed:          int
    total_analyzed:        int
    relevant_points:       int                  # failure points after architecture filter
    filtered_points:       int                  # failure points excluded by architecture
    architecture_summary:  Dict[str, Any] = {}  # snapshot of arch config used
    signal_summary:        Dict[str, int] = {}  # raw signal counts
    top_causes:            List[RCAFinding] = []


class AggregatedReport(BaseModel):
    run_id:           str
    project_id:       str
    agent_name:       str = ""    # human-readable name of the agent under test
    application_type: ApplicationType
    domain:           str
    timestamp:        datetime = Field(default_factory=datetime.utcnow)
    health_score:     float
    passed:           bool
    domain_weights:   Dict[str, float] = {}
    test_classes:     Dict[str, ClassSummary] = {}
    persona_breakdown: List[PersonaBreakdown] = []
    failure_drill_down: List[Dict[str, Any]] = []
    total_tests:      int
    total_passed:     int
    total_failed:     int
    total_skipped:    int = 0     # total skipped metric evaluations across all classes
    feedback_applied: bool = False
    evaluation_warnings: List[str] = []   # cross-class warnings surfaced to UI
    report_html:      str  = ""
    rca:              Optional[RCAReport] = None   # Stage 8 root cause analysis


# ── Stage 6: Feedback Loop ─────────────────────────────────────────────────

class PersonaFeedback(BaseModel):
    persona_id:       str
    persona_name:     str
    project_id:       str
    found_failures:   int
    total_runs:       int
    failure_rate:     float
    effective:        bool
    failure_patterns: List[str] = []
    suggested_action: str = "keep"   # strengthen | generate_variants | retire | keep


# ── Job Store Entry ────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    run_id:        str
    status:        str = "queued"   # queued | running | completed | failed
    progress:      int = 0          # 0-100
    message:       str = ""
    current_phase: str = ""
    phase_results: Dict[str, Any] = {}
    error:         Optional[str] = None
    results:       Optional[Dict[str, Any]] = None


# ── API Request/Response models ────────────────────────────────────────────

class PreviewRequest(BaseModel):
    agent_description: str
    agent_domain:      str = "general"
    application_type:  str = "chatbot"
    num_personas:      int = 3
    num_scenarios:     int = 5
    llm_provider:      str = "Groq"
    llm_api_key:       str = ""

class ConnectTestRequest(BaseModel):
    endpoint_url:         str
    request_field:        str = "message"
    response_field:       str = "response"
    auth_type:            str = "none"
    auth_token:           str = ""
    request_template:     Optional[str] = None
    response_trim_marker: Optional[str] = None

class ParseDocumentRequest(BaseModel):
    document_text:   str
    llm_provider:    str = "Groq"
    llm_api_key:     str = ""
