"""
Unified Pydantic schemas for the entire pipeline.
Sourced primarily from new metron-backend/app/core/models.py,
extended with fields from existing METRON TestConfig/TestResult dataclasses.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
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


# ── Stage 0b: Technical Profile (extracted from architecture doc) ──────────

class TechnicalProfile(BaseModel):
    """
    Generic technical intelligence extracted from any AI application document.
    24 categories covering every attack surface a red-team professional would
    look for — independent of the specific technology stack found.
    Populated by stages/s0_profile/technical_extractor.py.
    Consumed by attack_surface_mapper.py → security_gen.py + persona_builder.py.
    """
    # 1. LLM Context Structure — what reaches the model and in which role
    llm_context_messages: List[str] = []            # e.g. ["system", "user_metadata", "query"]
    user_controlled_in_system_prompt: List[str] = [] # user-supplied fields injected into system role
    injected_metadata_fields: List[str] = []         # e.g. ["team_id", "user_email", "role"]

    # 2. Authentication & Identity
    auth_type: str = ""                              # "jwt" | "api_key" | "oauth" | "session" | "none"
    auth_preprocessing: List[str] = []              # e.g. ["suffix_stripping", "base64_decode", "header_split"]
    identity_fields_in_context: List[str] = []      # identity fields passed to LLM e.g. ["user_id", "role"]
    identity_controls_behavior: bool = False         # does identity affect model selection or permissions

    # 3. Configuration Controls — runtime parameters that change system behavior
    config_controls_model: bool = False              # e.g. team_id → model selection
    config_parameters: List[str] = []               # config params that affect behavior
    user_influenced_config: List[str] = []          # config params user can influence or supply

    # 4. Tools and Capabilities — what the system can DO beyond answering
    authorized_actions: List[str] = []              # e.g. ["send_email", "read_calendar", "delete_file"]
    tool_invocation_model: str = ""                 # "llm_decided" | "rule_based" | "both"
    external_systems_accessible: List[str] = []     # e.g. ["smtp", "calendar_api", "sharepoint", "postgres"]

    # 5. Data Access Scope
    data_sources: List[str] = []                    # e.g. ["user_emails", "vector_db", "postgres"]
    retrieval_scope: str = ""                       # "per_user" | "shared" | "global" | "unknown"
    output_persisted: bool = False
    output_persistence_target: str = ""             # where LLM output is stored

    # 6. Output Destinations — where the response goes after generation
    output_destinations: List[str] = []             # e.g. ["email_body", "browser_render", "database", "downstream_api"]
    output_rendered_as: str = ""                    # "html" | "markdown" | "plain_text" | "json"

    # 7. Input Sources — what reaches the LLM besides the typed message
    additional_input_sources: List[str] = []        # e.g. ["email_headers", "file_upload", "db_lookup", "external_api"]
    external_data_injected_unsanitized: bool = False

    # 8. Logging and Observability
    logging_targets: List[str] = []                 # e.g. ["kafka", "cloudwatch", "postgres", "splunk"]
    llm_output_logged: bool = False
    log_searchable: bool = False                    # logs indexed and queryable

    # 9. Error and Fallback Behavior
    fallback_models: List[str] = []                 # weaker models used when primary fails
    error_info_exposed: List[str] = []              # info in error responses: "stack_trace" | "model_name" | "internal_url"
    fallback_path_exists: bool = False

    # 10. Hardcoded Values and Known Strings — literal secrets/tokens found in doc
    hardcoded_strings: List[str] = []               # literal tokens, keys, suffixes, passwords
    known_internal_identifiers: List[str] = []      # routing keys, team IDs, env var names

    # 11. Infrastructure and Deployment
    deployment_type: str = ""                       # "serverless" | "container" | "server" | "unknown"
    inter_service_trust: str = ""                   # "mutual_tls" | "shared_secret" | "none" | "unknown"
    execution_boundaries: List[str] = []            # e.g. ["lambda", "k8s_pod", "ecs_task"]

    # 12. Session and State
    session_persistence: List[str] = []             # where session stored e.g. ["redis", "postgres"]
    history_feeds_prompt: bool = False              # prior turns injected into future prompts
    cross_session_state: bool = False               # state persists across separate sessions

    # 13. Prompt Template Storage
    template_storage: str = ""                      # "hardcoded" | "database" | "config_file" | "cms" | "environment"
    template_modifiable_by: str = ""                # "admin" | "user" | "none" | "unknown"

    # 14. Multi-Tenancy and Isolation
    has_multi_tenancy: bool = False
    tenant_identifier: str = ""                     # e.g. "team_id", "org_id", "workspace_id"
    tenant_identifier_source: str = ""              # "jwt_claim" | "request_header" | "request_body" | "subdomain"
    isolation_mechanism: str = ""                   # "row_level_security" | "separate_db" | "filter_param" | "none"

    # 15. Caching
    response_caching: bool = False
    cache_key_fields: List[str] = []               # fields used to construct cache key
    cache_scope: str = ""                          # "per_user" | "shared" | "per_session" | "global"

    # 16. Content Filtering
    pre_llm_filters: List[str] = []               # filters applied BEFORE LLM call
    post_llm_filters: List[str] = []              # filters applied AFTER LLM response
    filter_implementation: str = ""               # "regex" | "classifier" | "llm_judge" | "external_api" | "blocklist"

    # 17. Human-in-the-Loop Controls
    has_hitl: bool = False
    hitl_trigger_threshold: str = ""              # what triggers human review (confidence score, category, etc.)
    hitl_approval_required: bool = False

    # 18. User Permission Model
    user_roles: List[str] = []                    # e.g. ["admin", "user", "guest", "internal"]
    role_validation_location: str = ""            # "server_side" | "client_side" | "middleware" | "unknown"
    role_controls_capabilities: bool = False

    # 19. Async and Queue Processing
    has_async_processing: bool = False
    queue_technology: str = ""                    # "kafka" | "sqs" | "rabbitmq" | "pubsub"
    async_skips_validation: bool = False          # async path has weaker validation than sync

    # 20. File and Document Processing
    accepted_file_types: List[str] = []          # e.g. ["pdf", "docx", "html", "csv", "txt"]
    file_parser_technology: str = ""             # "pypdf" | "docx" | "tika" | "custom"
    file_content_sanitized: bool = True          # False = file content injected raw into prompt

    # 21. Conversation History Retrieval
    history_retrieval_method: str = ""           # "all_turns" | "summary" | "vector_search" | "last_n"
    history_scope: str = ""                     # "per_user" | "per_session" | "per_conversation" | "global"
    history_injectable: bool = False            # history retrieval can be manipulated by user input

    # 22. Feedback and Ratings
    has_feedback_mechanism: bool = False
    feedback_influences_behavior: bool = False   # ratings affect future responses / fine-tuning
    feedback_stored_where: str = ""

    # 23. Compliance and Regulatory Scope
    compliance_frameworks: List[str] = []       # e.g. ["HIPAA", "GDPR", "SOC2", "PCI", "FERPA"]
    regulated_data_types: List[str] = []        # e.g. ["PII", "PHI", "PCI", "financial"]
    audit_trail_required: bool = False

    # 24. Output Post-Processing
    post_processing_steps: List[str] = []       # e.g. ["redaction", "truncation", "formatting", "translation"]
    redaction_implemented: bool = False
    output_format_enforced: str = ""            # "json" | "markdown" | "plain" | "html" | "none"

    # Summary generated by LLM extractor
    extraction_summary: str = ""


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

    @field_validator("ground_truth", mode="before")
    @classmethod
    def _validate_ground_truth(cls, v):
        """
        Validate ground_truth entries have the required 'question' key.
        Entries missing 'question' are dropped with a warning rather than
        crashing mid-pipeline with a cryptic KeyError.
        """
        if not isinstance(v, list):
            return []
        valid = []
        for i, entry in enumerate(v):
            if not isinstance(entry, dict):
                print(f"[RunConfig] ground_truth[{i}] is not a dict — skipped")
                continue
            if not entry.get("question", "").strip():
                print(f"[RunConfig] ground_truth[{i}] missing 'question' field — skipped")
                continue
            valid.append(entry)
        return valid

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
    # deepeval_metrics controls which DeepEval metrics run in functional evaluation:
    #   "hallucination"    — HallucinationMetric (requires context or expected_behavior)
    #   "answer_relevancy" — AnswerRelevancyMetric
    # Note: toxicity is evaluated separately in security.py via Detoxify (always on).
    ragas_metrics:    List[str] = ["faithfulness", "answer_relevancy"]
    deepeval_metrics: List[str] = ["hallucination", "answer_relevancy"]
    use_geval:        bool = True

    # Adapter timeout — how long to wait for the target endpoint before giving up.
    # Increase for slow LLM inference backends (e.g. self-hosted models).
    adapter_timeout: int = 60   # seconds

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
    entry_points:        List[str] = []
    initial_state:       ConversationState = ConversationState.SEEKING
    fishbone_dimensions: Dict[str, str] = {}
    # Rich fields from testing taxonomy (populated by upgraded persona_builder)
    testing_taxonomy_ids: List[str] = []     # e.g. ["A01", "A08"] or ["U07"]
    edge_case_taxonomy_id: str = ""          # U01-U08 for user/edge_case personas
    attack_trajectory:   List[Dict[str, Any]] = []   # 5-turn adversarial conversation trajectory
    playbook_steps:      List[Dict[str, Any]] = []   # 4+ step attack playbook with literal prompts
    multi_turn_scenario: List[Dict[str, Any]] = []   # 3-turn user edge-case scenario with literal prompts
    evasion_techniques:  List[str] = []     # embedded evasion techniques (adversarial only)
    risk_severity:       str = ""           # critical|high|medium|low (adversarial only)


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
    # Testing taxonomy enrichment (from core/testing_taxonomy.py)
    mitre_atlas_id:         Optional[str] = None   # e.g. "AML.T0051"


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
