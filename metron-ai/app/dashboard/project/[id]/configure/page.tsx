"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { authFetch } from "@/lib/api";

const API = "";

const DOMAINS = [
  "General", "Customer Support", "E-commerce", "Healthcare", "Finance",
  "Education", "Technical Support", "HR/Recruitment", "Legal", "Travel", "Other",
];

const APPLICATION_TYPES = [
  { value: "chatbot", label: "Chatbot",   icon: "chat",          description: "Generic JSON REST chatbot" },
  { value: "rag",     label: "RAG Agent", icon: "library_books", description: "Retrieval-augmented generation" },
];


interface Provider {
  description: string;
  rpm: number;
  models: Record<string, string>;
  default: string;
  env_key: string;
}

export default function ConfigurePage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  // Pre-populated from dashboard modal
  const [endpointUrl, setEndpointUrl] = useState("");
  const [requestField, setRequestField] = useState("message");
  const [responseField, setResponseField] = useState("response");
  const [authType, setAuthType] = useState<"none" | "bearer">("none");
  const [authToken, setAuthToken] = useState("");
  const [requestTemplate, setRequestTemplate] = useState("");
  const [responseTrimMarker, setResponseTrimMarker] = useState("");
  const [connectionStatus, setConnectionStatus] = useState<"idle" | "testing" | "ok" | "fail">("idle");
  const [connectionMsg, setConnectionMsg] = useState("");

  // Agent
  const [agentName, setAgentName] = useState("");
  const [agentDomain, setAgentDomain] = useState("General");
  const [agentDescription, setAgentDescription] = useState("");

  // RAG
  const [isRag, setIsRag] = useState(false);
  const [groundTruthFile, setGroundTruthFile] = useState<File | null>(null);

  // Test params
  const [numPersonas, setNumPersonas] = useState(3);
  const [numScenarios, setNumScenarios] = useState(5);
  const [convTurns, setConvTurns] = useState(3);
  const [enableJudge, setEnableJudge] = useState(true);
  const [perfRequests, setPerfRequests] = useState(20);
  const [loadUsers, setLoadUsers] = useState(5);
  const [loadDuration, setLoadDuration] = useState(30);

  // LLM
  const [providers, setProviders] = useState<Record<string, Provider>>({});
  const [llmProvider, setLlmProvider] = useState("NVIDIA NIM");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [azureEndpoint, setAzureEndpoint] = useState("");

  // Security
  const [selectedAttacks, setSelectedAttacks] = useState<string[]>(["jailbreak", "prompt_injection", "pii_extraction", "toxicity", "encoding"]);
  const [attacksPerCategory, setAttacksPerCategory] = useState(3);

  // Quality metrics
  const [ragasMetrics, setRagasMetrics] = useState<string[]>(["faithfulness", "answer_relevancy"]);
  const [deepevalMetrics, setDeepevalMetrics] = useState<string[]>(["hallucination", "toxicity"]);
  const [useGeval, setUseGeval] = useState(false);

  const [isNavigating, setIsNavigating] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Application type
  const [applicationType, setApplicationType] = useState("chatbot");

  // Architecture profile (for RCA)
  const [deploymentType, setDeploymentType]           = useState("unknown");
  const [vectorDb, setVectorDb]                       = useState("");
  const [sessionDb, setSessionDb]                     = useState("");
  const [cacheLayer, setCacheLayer]                   = useState("");
  const [messageQueue, setMessageQueue]               = useState("");
  const [apiGateway, setApiGateway]                   = useState("");
  const [authMechanism, setAuthMechanism]             = useState("");
  const [monitoringTool, setMonitoringTool]           = useState("");
  const [isMultiRegion, setIsMultiRegion]             = useState(false);
  const [hasRateLimiting, setHasRateLimiting]         = useState(false);
  const [hasRetryLogic, setHasRetryLogic]             = useState(false);
  const [hasCircuitBreaker, setHasCircuitBreaker]     = useState(false);
  const [hasCaching, setHasCaching]                   = useState(false);
  const [hasDlq, setHasDlq]                           = useState(false);
  const [archNotes, setArchNotes]                     = useState("");
  // Architecture upload / auto-extract
  const [archTab, setArchTab]                         = useState<"form" | "upload">("form");
  const [archDocFile, setArchDocFile]                 = useState<File | null>(null);
  const [archDiagramFile, setArchDiagramFile]         = useState<File | null>(null);
  const [archExtracting, setArchExtracting]           = useState(false);
  const [archExtractMsg, setArchExtractMsg]           = useState("");


  // Persona / scenario generation
  interface Persona { id: string; name: string; description: string; traits: string[]; sample_prompts: string[] }
  interface Scenario { id: string; name: string; description: string; initial_prompt: string; expected_behavior: string; category: string }
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState("");
  const [expandedPersona, setExpandedPersona] = useState<string | null>(null);

  // Load initial config from sessionStorage
  useEffect(() => {
    const stored = sessionStorage.getItem(`project_${projectId}`);
    if (stored) {
      try {
        const data = JSON.parse(stored);
        if (data.endpoint) setEndpointUrl(data.endpoint);
        if (data.apiKey) {
          setAuthType("bearer");
          setAuthToken(data.apiKey);
        }
        // Knowledge base textarea removed — ground truth file is the only RAG input.
        // The uploaded document is the system description for persona generation.
        // The knowledge base textarea is a separate optional field for RAG faithfulness scoring.
        if (data.name) setAgentName(data.name);
      } catch (_) {}
    }
  }, [projectId]);

  // Load providers
  useEffect(() => {
    authFetch(`${API}/api/providers`)
      .then((r) => r.json())
      .then((data) => setProviders(data))
      .catch(() => {});
  }, []);

  const testConnection = async () => {
    if (!endpointUrl) return;
    setConnectionStatus("testing");
    try {
      const res = await authFetch(`${API}/api/connect-test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint_url: endpointUrl,
          request_field: requestField,
          response_field: responseField,
          auth_type: authType,
          auth_token: authToken,
          request_template: requestTemplate || null,
          response_trim_marker: responseTrimMarker || null,
        }),
      });
      const data = await res.json();
      setConnectionStatus(data.success ? "ok" : "fail");
      setConnectionMsg(data.message);
    } catch (e) {
      setConnectionStatus("fail");
      setConnectionMsg("Could not reach backend");
    }
  };


  const handleGenerate = async () => {
    if (!agentDescription) {
      setErrors({ description: "Agent description is required to generate personas" });
      return;
    }
    setErrors({});
    setIsGenerating(true);
    setGenerateError("");
    try {
      const res = await authFetch(`${API}/api/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_description: agentDescription,
          agent_domain: agentDomain,
          application_type: applicationType,
          num_personas: numPersonas,
          num_scenarios: numScenarios,
          llm_provider: llmProvider,
          llm_api_key: llmApiKey,
          azure_endpoint: azureEndpoint,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error((err as { detail: string }).detail || "Generation failed");
      }
      const data = await res.json();
      setPersonas(data.personas || []);
      setScenarios(data.scenarios || []);
      // Cache for preview page too
      sessionStorage.setItem(`preview_${projectId}`, JSON.stringify(data));
    } catch (e: unknown) {
      setGenerateError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleNext = async () => {
    const errs: Record<string, string> = {};
    if (!endpointUrl) errs.endpoint = "Endpoint URL is required";
    if (!agentDescription) errs.description = "Agent description is required";
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      return;
    }
    setErrors({});
    setIsNavigating(true);

    // Step 1: Read the ground truth file to completion BEFORE building the config.
    // FileReader is callback-based; wrapping it in a Promise lets us await it so
    // that both sessionStorage and rag_text are populated with real data rather
    // than whatever was left over from a previous run.
    let groundTruthText = "";
    if (isRag && groundTruthFile) {
      groundTruthText = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror  = () => resolve("");
        reader.readAsText(groundTruthFile);
      });
      sessionStorage.setItem(`ground_truth_${projectId}`, groundTruthText);
      sessionStorage.setItem(`ground_truth_name_${projectId}`, groundTruthFile.name);
    } else {
      sessionStorage.removeItem(`ground_truth_${projectId}`);
      sessionStorage.removeItem(`ground_truth_name_${projectId}`);
    }

    // Step 2: Build full config — rag_text is now correct because groundTruthText
    // was already loaded above.  Only the first 800 chars are sent as a context
    // hint for persona generation; the full file is uploaded separately at run-time.
    const fullConfig = {
      project_id: projectId,
      endpoint_url: endpointUrl,
      request_field: requestField,
      response_field: responseField,
      auth_type: authType,
      auth_token: authToken,
      request_template: requestTemplate || null,
      response_trim_marker: responseTrimMarker || null,
      agent_name: agentName,
      agent_domain: agentDomain,
      agent_description: agentDescription,
      is_rag: isRag,
      rag_text: isRag ? groundTruthText.slice(0, 800) : "",
      num_personas: numPersonas,
      num_scenarios: numScenarios,
      conversation_turns: convTurns,
      enable_judge: enableJudge,
      performance_requests: perfRequests,
      load_concurrent_users: loadUsers,
      load_duration_seconds: loadDuration,
      llm_provider: llmProvider,
      llm_api_key: llmApiKey,
      azure_endpoint: azureEndpoint,
      selected_attacks: selectedAttacks,
      attacks_per_category: attacksPerCategory,
      ragas_metrics: ragasMetrics,
      deepeval_metrics: deepevalMetrics,
      use_geval: useGeval,
      application_type: applicationType,
      deployment_type: deploymentType,
      vector_db: vectorDb,
      session_db: sessionDb,
      cache_layer: cacheLayer,
      message_queue: messageQueue,
      api_gateway: apiGateway,
      auth_mechanism: authMechanism,
      monitoring_tool: monitoringTool,
      is_multi_region: isMultiRegion,
      has_rate_limiting: hasRateLimiting,
      has_retry_logic: hasRetryLogic,
      has_circuit_breaker: hasCircuitBreaker,
      has_caching: hasCaching,
      has_dlq: hasDlq,
      additional_architecture_notes: archNotes,
    };

    sessionStorage.setItem(`fullconfig_${projectId}`, JSON.stringify(fullConfig));

    // Step 3: Generate personas/scenarios only if not already done
    if (personas.length === 0) {
      try {
        const res = await authFetch(`${API}/api/preview`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            agent_description: agentDescription,
            agent_domain: agentDomain,
            application_type: applicationType,
            num_personas: numPersonas,
            num_scenarios: numScenarios,
            llm_provider: llmProvider,
            llm_api_key: llmApiKey,
            azure_endpoint: azureEndpoint,
          }),
        });
        if (res.ok) {
          const preview = await res.json();
          setPersonas(preview.personas || []);
          setScenarios(preview.scenarios || []);
          sessionStorage.setItem(`preview_${projectId}`, JSON.stringify(preview));
        }
      } catch (_) {}
    }

    router.push(`/dashboard/project/${projectId}/preview`);
  };

  const providerInfo = providers[llmProvider];

  // ── Architecture auto-extract ──────────────────────────────────────────
  const handleArchExtract = async () => {
    if (!archDocFile && !archDiagramFile) return;
    setArchExtracting(true);
    setArchExtractMsg("");
    try {
      const fd = new FormData();
      // Backend expects content as plain text string, not a file
      if (archDocFile) {
        const docText = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload  = () => resolve(reader.result as string);
          reader.onerror = () => resolve("");
          reader.readAsText(archDocFile);
        });
        fd.append("content", docText);
      }
      if (archDiagramFile) fd.append("image", archDiagramFile, archDiagramFile.name);
      fd.append("llm_provider",   llmProvider);
      fd.append("llm_api_key",    llmApiKey);
      fd.append("azure_endpoint", azureEndpoint);

      const res = await authFetch(`${API}/api/parse-architecture`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      if (data.deployment_type)  setDeploymentType(data.deployment_type);
      if (data.vector_db)        setVectorDb(data.vector_db);
      if (data.session_db)       setSessionDb(data.session_db);
      if (data.cache_layer)      setCacheLayer(data.cache_layer);
      if (data.message_queue)    setMessageQueue(data.message_queue);
      if (data.api_gateway)      setApiGateway(data.api_gateway);
      if (data.auth_mechanism)   setAuthMechanism(data.auth_mechanism);
      if (data.monitoring_tool)  setMonitoringTool(data.monitoring_tool);
      if (typeof data.is_multi_region   === "boolean") setIsMultiRegion(data.is_multi_region);
      if (typeof data.has_rate_limiting === "boolean") setHasRateLimiting(data.has_rate_limiting);
      if (typeof data.has_retry_logic   === "boolean") setHasRetryLogic(data.has_retry_logic);
      if (typeof data.has_circuit_breaker === "boolean") setHasCircuitBreaker(data.has_circuit_breaker);
      if (typeof data.has_caching       === "boolean") setHasCaching(data.has_caching);
      if (typeof data.has_dlq           === "boolean") setHasDlq(data.has_dlq);
      if (data.summary) setArchNotes((prev) => prev ? prev : data.summary);

      setArchExtractMsg("Fields auto-filled from document. Review and adjust below.");
      setArchTab("form");
    } catch (e: unknown) {
      setArchExtractMsg(e instanceof Error ? e.message : "Extraction failed");
    } finally {
      setArchExtracting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto pb-20 space-y-10 animate-fade-in">
      {/* Header */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-primary" />
          <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">Step 1 of 3</span>
        </div>
        <h1 className="font-headline text-4xl font-black text-[var(--color-on-surface)] tracking-tighter">Configure Test Suite</h1>
        <p className="text-[var(--color-on-surface-variant)] text-sm font-medium opacity-60">Set up your endpoint, agent details, and test parameters.</p>
      </div>

      {/* ── Endpoint Config ─────────────────────────────────── */}
      <Section title="Endpoint Configuration" icon="link">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-4">
            <Field label="API Endpoint URL" error={errors.endpoint}>
              <input
                className="input-field"
                placeholder="https://your-api.com/chat"
                value={endpointUrl}
                onChange={(e) => setEndpointUrl(e.target.value)}
              />
            </Field>
            <Field label="Request Field">
              <input className="input-field" placeholder="message" value={requestField} onChange={(e) => setRequestField(e.target.value)} />
              <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-50 mt-1">Ignored when Request Template is set.</p>
            </Field>
            <Field label="Response Field (dot-notation)">
              <input className="input-field" placeholder="response  or  result.artifacts.0.parts.0.text" value={responseField} onChange={(e) => setResponseField(e.target.value)} />
            </Field>
            <Field label="Request Template (optional — for complex APIs)">
              <textarea
                className="input-field resize-none h-[110px] font-mono text-xs"
                placeholder={'{\n  "id": "{{uuid}}",\n  "message": "{{query}}",\n  "sessionId": "{{conversation_id}}"\n}'}
                value={requestTemplate}
                onChange={(e) => setRequestTemplate(e.target.value)}
              />
              <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-50 mt-1">
                Full JSON body. Use <code>{"{{query}}"}</code> for the message, <code>{"{{uuid}}"}</code> for a per-request UUID, <code>{"{{conversation_id}}"}</code> for a per-conversation UUID (stable across multi-turn).
              </p>
            </Field>
            <Field label="Response Trim Marker (optional)">
              <input
                className="input-field font-mono text-xs"
                placeholder="FOLLOW UP QUESTIONS"
                value={responseTrimMarker}
                onChange={(e) => setResponseTrimMarker(e.target.value)}
              />
              <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-50 mt-1">
                Text at or after this marker is stripped from every response before evaluation.
              </p>
            </Field>
          </div>
          <div className="space-y-4">
            <Field label="Authentication">
              <select className="input-field" aria-label="Authentication" value={authType} onChange={(e) => setAuthType(e.target.value as "none" | "bearer")}>
                <option value="none">None</option>
                <option value="bearer">Bearer Token</option>
              </select>
            </Field>
            {authType === "bearer" && (
              <Field label="Bearer Token">
                <input className="input-field" type="password" placeholder="••••••••" value={authToken} onChange={(e) => setAuthToken(e.target.value)} />
              </Field>
            )}
            <div className="pt-2">
              <button
                onClick={testConnection}
                disabled={!endpointUrl || connectionStatus === "testing"}
                className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl border border-[var(--color-outline)] text-sm font-semibold hover:bg-[var(--color-surface-variant)] transition-colors disabled:opacity-40"
              >
                <span className="material-symbols-outlined text-base">
                  {connectionStatus === "testing" ? "sync" : connectionStatus === "ok" ? "check_circle" : connectionStatus === "fail" ? "error" : "lan"}
                </span>
                {connectionStatus === "testing" ? "Testing…" : "Test Connection"}
              </button>
              {connectionMsg && (
                <p className={`mt-2 text-xs ${connectionStatus === "ok" ? "text-secondary" : "text-error"}`}>{connectionMsg}</p>
              )}
            </div>
          </div>
        </div>
      </Section>

      {/* ── Application Type ─────────────────────────────── */}
      <Section title="Application Type" icon="apps">
        <p className="text-sm text-[var(--color-on-surface-variant)] opacity-70 -mt-2">
          Select the type that best describes your AI system. This guides test generation and adapter selection.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {APPLICATION_TYPES.map((type) => (
            <button
              key={type.value}
              onClick={() => setApplicationType(type.value)}
              className={`flex flex-col items-center gap-2 p-4 rounded-xl border-2 text-center transition-all ${
                applicationType === type.value
                  ? "border-primary bg-primary/5"
                  : "border-[var(--color-outline-variant)] hover:border-primary/50"
              }`}
            >
              <span className={`material-symbols-outlined text-2xl ${applicationType === type.value ? "text-primary" : "text-[var(--color-on-surface-variant)]"}`}>
                {type.icon}
              </span>
              <span className={`text-sm font-bold ${applicationType === type.value ? "text-primary" : "text-[var(--color-on-surface)]"}`}>
                {type.label}
              </span>
              <span className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60 leading-tight">
                {type.description}
              </span>
            </button>
          ))}
        </div>
      </Section>

      {/* ── Agent Description ─────────────────────────────── */}
      <Section title="Agent Description" icon="smart_toy">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-4">
            <Field label="Agent Name">
              <input className="input-field" placeholder="Customer Support Bot" value={agentName} onChange={(e) => setAgentName(e.target.value)} />
            </Field>
            <Field label="Domain">
              <select className="input-field" aria-label="Domain" value={agentDomain} onChange={(e) => setAgentDomain(e.target.value)}>
                {DOMAINS.map((d) => <option key={d}>{d}</option>)}
              </select>
            </Field>
          </div>
          <Field label="Agent Description" error={errors.description}>
            <textarea
              className="input-field resize-none h-[120px]"
              placeholder="Describe what your agent does, its capabilities, expected behavior, and any constraints…"
              value={agentDescription}
              onChange={(e) => setAgentDescription(e.target.value)}
            />
          </Field>
        </div>

        {/* Generate Personas button */}
        <div className="pt-2 flex items-center gap-4">
          <button
            onClick={handleGenerate}
            disabled={!agentDescription || isGenerating}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-primary text-primary text-sm font-semibold hover:bg-primary/5 transition-colors disabled:opacity-40"
          >
            <span className={`material-symbols-outlined text-base ${isGenerating ? "animate-spin" : ""}`}>
              {isGenerating ? "progress_activity" : "group_add"}
            </span>
            {isGenerating ? "Generating…" : personas.length > 0 ? `Regenerate Personas (${personas.length})` : "Generate Personas & Scenarios"}
          </button>
          {personas.length > 0 && !isGenerating && (
            <span className="text-xs text-secondary font-semibold flex items-center gap-1">
              <span className="material-symbols-outlined text-sm">check_circle</span>
              {personas.length} personas · {scenarios.length} scenarios ready
            </span>
          )}
        </div>

        {generateError && (
          <p className="text-xs text-error flex items-center gap-1 mt-1">
            <span className="material-symbols-outlined text-sm">error</span>
            {generateError}
          </p>
        )}

        {/* Persona cards */}
        {personas.length > 0 && (
          <div className="mt-4 space-y-2">
            <p className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">
              Generated Personas
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {personas.map((p) => (
                <div
                  key={p.id}
                  className="border border-[var(--color-outline-variant)] rounded-xl overflow-hidden"
                >
                  <button
                    onClick={() => setExpandedPersona(expandedPersona === p.id ? null : p.id)}
                    className="w-full p-3 text-left hover:bg-[var(--color-surface-container-low)] transition-colors"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-black leading-tight">{p.name}</p>
                      <span className="material-symbols-outlined text-sm text-[var(--color-on-surface-variant)] flex-shrink-0">
                        {expandedPersona === p.id ? "expand_less" : "expand_more"}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60 mt-1 line-clamp-2">
                      {p.description}
                    </p>
                  </button>

                  {expandedPersona === p.id && (
                    <div className="px-3 pb-3 space-y-2 border-t border-[var(--color-outline-variant)]">
                      {p.traits.length > 0 && (
                        <div className="pt-2">
                          <p className="text-[10px] font-bold uppercase tracking-wider opacity-50 mb-1">Traits</p>
                          <div className="flex flex-wrap gap-1">
                            {p.traits.map((t) => (
                              <span key={t} className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
                                {t}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {p.sample_prompts.slice(0, 2).map((prompt, i) => (
                        <p key={i} className="text-xs italic text-[var(--color-on-surface-variant)] pl-2 border-l-2 border-primary/30">
                          "{prompt}"
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </Section>

      {/* ── RAG Config ─────────────────────────────────────── */}
      <Section title="RAG Configuration" icon="library_books">
        <label className="flex items-center gap-3 cursor-pointer mb-4">
          <div
            onClick={() => setIsRag(!isRag)}
            className={`w-10 h-6 rounded-full transition-colors relative ${isRag ? "bg-primary" : "bg-[var(--color-outline)]"}`}
          >
            <span className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-transform ${isRag ? "translate-x-5" : "translate-x-1"}`} />
          </div>
          <span className="text-sm font-semibold text-[var(--color-on-surface)]">This is a RAG Agent</span>
        </label>
        {isRag && (
          <div className="space-y-5">
            <Field label="Ground Truth File — question, expected_answer & context (CSV or JSON)">
              <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70 mb-3">
                Each row/object must have <strong>question</strong>, <strong>expected_answer</strong>, and <strong>context</strong> fields.
                Context is used as the sole source of truth for faithfulness, recall &amp; precision scoring.
              </p>
              <div className="relative group p-5 border-2 border-dashed border-[var(--color-outline-variant)] border-opacity-30 rounded-2xl bg-[var(--color-surface-container-low)]/50 hover:border-primary/50 hover:bg-primary/5 transition-all text-center cursor-pointer">
                <input
                  type="file"
                  accept=".csv,.json"
                  aria-label="Upload Ground Truth File"
                  className="absolute inset-0 opacity-0 cursor-pointer z-10"
                  onChange={(e) => setGroundTruthFile(e.target.files?.[0] || null)}
                />
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2 transition-all ${groundTruthFile ? "bg-[#6bff8f] text-[#006e2f]" : "bg-primary/10 text-primary"}`}>
                  <span className="material-symbols-outlined text-xl">{groundTruthFile ? "task_alt" : "upload_file"}</span>
                </div>
                {groundTruthFile ? (
                  <div>
                    <p className="font-bold text-sm text-on-surface truncate">{groundTruthFile.name}</p>
                    <p className="text-[10px] font-bold text-[#006e2f] uppercase tracking-widest mt-0.5">Ready</p>
                  </div>
                ) : (
                  <div>
                    <p className="text-sm font-semibold text-on-surface">Upload ground truth file</p>
                    <p className="text-[9px] font-bold uppercase tracking-widest text-outline opacity-50 mt-0.5">
                      CSV: question, expected_answer, context &nbsp;·&nbsp; JSON: [{"{"}question, expected_answer, context{"}"}]
                    </p>
                  </div>
                )}
              </div>
            </Field>
          </div>
        )}
      </Section>

      {/* ── Test Parameters ────────────────────────────────── */}
      <Section title="Test Parameters" icon="tune">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-4">
            <SliderField label="Personas" min={1} max={50} value={numPersonas} onChange={setNumPersonas} />
            <SliderField label="Scenarios" min={1} max={50} value={numScenarios} onChange={setNumScenarios} />
          </div>
          <div className="space-y-4">
            <SliderField label="Conversation Turns" min={1} max={15} value={convTurns} onChange={setConvTurns} />
            <label className="flex items-center gap-3 cursor-pointer">
              <div
                onClick={() => setEnableJudge(!enableJudge)}
                className={`w-10 h-6 rounded-full transition-colors relative ${enableJudge ? "bg-primary" : "bg-[var(--color-outline)]"}`}
              >
                <span className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-transform ${enableJudge ? "translate-x-5" : "translate-x-1"}`} />
              </div>
              <div>
                <p className="text-sm font-semibold text-[var(--color-on-surface)]">Enable LLM Judge</p>
                <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">More accurate, slower</p>
              </div>
            </label>
          </div>
          <div className="space-y-4">
            <SliderField label="Performance Requests" min={5} max={1000} value={perfRequests} onChange={setPerfRequests} />
            <SliderField label="Load Test Users" min={1} max={1000} value={loadUsers} onChange={setLoadUsers} />
            <SliderField label="Load Duration (s)" min={10} max={1000} value={loadDuration} onChange={setLoadDuration} />
          </div>
        </div>
      </Section>

      {/* ── LLM Provider ──────────────────────────────────── */}
      <Section title="LLM Provider" icon="psychology">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-4">
            <Field label="Provider">
              <select className="input-field" aria-label="Provider" value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)}>
                {Object.keys(providers).length > 0
                  ? Object.keys(providers).map((p) => <option key={p}>{p}</option>)
                  : ["NVIDIA NIM", "Groq", "Google Gemini", "Azure OpenAI"].map((p) => <option key={p}>{p}</option>)}
              </select>
            </Field>
            {providerInfo && (
              <div className="p-3 rounded-xl bg-[var(--color-surface-variant)] space-y-1">
                <p className="text-xs font-semibold text-[var(--color-on-surface)]">{providerInfo.description}</p>
                <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">{providerInfo.rpm} RPM</p>
              </div>
            )}
          </div>
          <div className="space-y-4">
            <Field label="API Key">
              <input
                className="input-field"
                type="password"
                placeholder="Enter your API key (or set env variable)"
                value={llmApiKey}
                onChange={(e) => setLlmApiKey(e.target.value)}
              />
            </Field>
            {llmProvider === "Azure OpenAI" && (
              <Field label="Azure Endpoint URL">
                <input
                  className="input-field"
                  type="text"
                  placeholder="https://your-resource.openai.azure.com/"
                  value={azureEndpoint}
                  onChange={(e) => setAzureEndpoint(e.target.value)}
                />
              </Field>
            )}
            {providerInfo && (
              <div className="p-3 rounded-xl bg-[var(--color-surface-variant)] space-y-1">
                <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">Default model</p>
                <p className="text-xs font-mono font-semibold text-[var(--color-on-surface)]">{providerInfo.default}</p>
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* ── Functional Tests ──────────────────────────────── */}
      <Section title="Functional Tests" icon="science">
        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70 -mt-2">Run on every functional conversation to measure core response quality.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {[
            { label: "Hallucination",    icon: "psychology_alt", note: "DeepEval HallucinationMetric" },
            { label: "Answer Relevancy", icon: "ads_click",      note: "DeepEval AnswerRelevancyMetric" },
            { label: "Usefulness",       icon: "thumb_up",       note: "DeepEval GEval" },
            { label: "LLM Judge",        icon: "gavel",          note: "Relevance · Accuracy · Helpfulness (domain-specific)" },
          ].map((t) => (
            <div key={t.label} className="flex items-center gap-3 p-3 rounded-xl bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
              <span className="material-symbols-outlined text-base text-primary">{t.icon}</span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-[var(--color-on-surface)]">{t.label}</p>
                <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">{t.note}</p>
              </div>
              <span className="material-symbols-outlined text-sm text-secondary ml-auto">check_circle</span>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Security Tests ─────────────────────────────────── */}
      <Section title="Security Tests" icon="security">
        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70 -mt-2">These evaluations always run on every conversation.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-5">
          {[
            { label: "Prompt Injection",  icon: "code_blocks",  note: "LLM Guard" },
            { label: "PII Leakage",       icon: "policy",       note: "Presidio" },
            { label: "Toxicity (Output)", icon: "block",        note: "Detoxify" },
            { label: "Bias & Fairness",   icon: "balance",      note: "DeepEval" },
            { label: "Toxic Request",     icon: "dangerous",    note: "Golden dataset + Detoxify" },
            { label: "Attack Resistance", icon: "shield",       note: "LLM Judge" },
          ].map((t) => (
            <div key={t.label} className="flex items-center gap-3 p-3 rounded-xl bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
              <span className="material-symbols-outlined text-base text-secondary">{t.icon}</span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-[var(--color-on-surface)]">{t.label}</p>
                <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">{t.note}</p>
              </div>
              <span className="material-symbols-outlined text-sm text-secondary ml-auto">check_circle</span>
            </div>
          ))}
        </div>
        <SliderField label="Attack Prompts per Category" min={1} max={20} value={attacksPerCategory} onChange={setAttacksPerCategory} />
      </Section>

      {/* ── Quality Metrics ────────────────────────────────── */}
      <Section title="Quality Metrics" icon="grade">
        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70 -mt-2">These metrics run on all functional and quality conversations. Security conversations are excluded to avoid content-filter conflicts.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-4">
          {[
            { label: "Hallucination",    note: "DeepEval" },
            { label: "Answer Relevancy", note: "DeepEval" },
            { label: "Usefulness",       note: "DeepEval" },
          ].map((m) => (
            <div key={m.label} className="flex items-center gap-3 p-3 rounded-xl bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
              <span className="material-symbols-outlined text-base text-secondary">check_circle</span>
              <div>
                <p className="text-sm font-semibold text-[var(--color-on-surface)]">{m.label}</p>
                <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">{m.note}</p>
              </div>
            </div>
          ))}
          <div className="flex items-center gap-3 p-3 rounded-xl bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
            <span className="material-symbols-outlined text-base text-secondary">check_circle</span>
            <div>
              <p className="text-sm font-semibold text-[var(--color-on-surface)]">GEval (Domain Criteria)</p>
              <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">DeepEval — auto-generated for your domain</p>
            </div>
          </div>
        </div>
        {isRag && (
          <div className="space-y-2">
            <p className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">RAG Mode — runs when retrieved context is available</p>
            {[
              { label: "Faithfulness",      note: "RAGAS" },
              { label: "Context Recall",    note: "RAGAS" },
              { label: "Context Precision", note: "RAGAS" },
            ].map((m) => (
              <div key={m.label} className="flex items-center gap-3 p-3 rounded-xl bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
                <span className="material-symbols-outlined text-base text-secondary">check_circle</span>
                <div>
                  <p className="text-sm font-semibold text-[var(--color-on-surface)]">{m.label}</p>
                  <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">{m.note}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* ── Load & Performance ────────────────────────────── */}
      <Section title="Load & Performance" icon="speed">
        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70 -mt-2">Measures response time and throughput. No external tools required — built-in async HTTP timing.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {[
            { label: "Response Latency",  icon: "timer",         note: "avg · p50 · p95 · p99 ms" },
            { label: "Throughput",        icon: "trending_up",   note: "Requests per second" },
            { label: "Error Rate",        icon: "error_outline", note: "% failed requests" },
            { label: "Concurrent Load",   icon: "group",         note: `${loadUsers} virtual users · ${loadDuration}s` },
          ].map((t) => (
            <div key={t.label} className="flex items-center gap-3 p-3 rounded-xl bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
              <span className="material-symbols-outlined text-base text-[var(--color-tertiary,#855300)]">{t.icon}</span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-[var(--color-on-surface)]">{t.label}</p>
                <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">{t.note}</p>
              </div>
              <span className="material-symbols-outlined text-sm text-secondary ml-auto">check_circle</span>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Architecture Profile (RCA) ───────────────────── */}
      <Section title="Architecture Profile" icon="account_tree">
        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70 -mt-2">
          Optional — helps the Root Cause Analysis engine filter and prioritise the most relevant failure points for your specific infrastructure.
        </p>

        {/* Tab switcher */}
        <div className="flex gap-1 p-1 rounded-xl bg-[var(--color-surface-variant)] w-fit">
          {(["form", "upload"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setArchTab(tab)}
              className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${
                archTab === tab
                  ? "bg-[var(--color-surface)] shadow text-[var(--color-on-surface)]"
                  : "text-[var(--color-on-surface-variant)] opacity-60 hover:opacity-100"
              }`}
            >
              <span className="material-symbols-outlined text-sm">
                {tab === "form" ? "list_alt" : "upload_file"}
              </span>
              {tab === "form" ? "Structured Form" : "Upload Doc / Diagram"}
            </button>
          ))}
        </div>

        {archExtractMsg && (
          <div className={`flex items-center gap-2 p-3 rounded-xl text-xs font-semibold ${archExtractMsg.includes("auto-filled") ? "bg-secondary/10 text-secondary" : "bg-error/10 text-error"}`}>
            <span className="material-symbols-outlined text-sm">{archExtractMsg.includes("auto-filled") ? "check_circle" : "error"}</span>
            {archExtractMsg}
          </div>
        )}

        {/* ── Form tab ────────────────────────────────────── */}
        {archTab === "form" && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Field label="Deployment Type">
                <select className="input-field" aria-label="Deployment Type" value={deploymentType} onChange={(e) => setDeploymentType(e.target.value)}>
                  <option value="unknown">Unknown / Not sure</option>
                  <option value="serverless">Serverless (Lambda, Azure Functions, Cloud Run)</option>
                  <option value="server">Server / VM (always-on)</option>
                  <option value="container">Container (Kubernetes, Docker)</option>
                </select>
              </Field>

              <Field label="Vector DB (if RAG)">
                <select className="input-field" aria-label="Vector DB" value={vectorDb} onChange={(e) => setVectorDb(e.target.value)}>
                  <option value="">None / Not applicable</option>
                  <option value="pinecone">Pinecone</option>
                  <option value="weaviate">Weaviate</option>
                  <option value="qdrant">Qdrant</option>
                  <option value="faiss">FAISS</option>
                  <option value="chroma">Chroma</option>
                  <option value="other">Other</option>
                </select>
              </Field>

              <Field label="Session / Memory DB">
                <select className="input-field" aria-label="Session / Memory DB" value={sessionDb} onChange={(e) => setSessionDb(e.target.value)}>
                  <option value="">None / Stateless</option>
                  <option value="redis">Redis</option>
                  <option value="postgresql">PostgreSQL</option>
                  <option value="mongodb">MongoDB</option>
                  <option value="dynamodb">DynamoDB</option>
                  <option value="sqlite">SQLite</option>
                  <option value="other">Other</option>
                </select>
              </Field>

              <Field label="Cache Layer">
                <select className="input-field" aria-label="Cache Layer" value={cacheLayer} onChange={(e) => setCacheLayer(e.target.value)}>
                  <option value="">None</option>
                  <option value="redis">Redis</option>
                  <option value="memcached">Memcached</option>
                  <option value="cdn">CDN</option>
                  <option value="in_memory">In-memory</option>
                  <option value="other">Other</option>
                </select>
              </Field>

              <Field label="Message Queue">
                <select className="input-field" aria-label="Message Queue" value={messageQueue} onChange={(e) => setMessageQueue(e.target.value)}>
                  <option value="">None</option>
                  <option value="kafka">Kafka</option>
                  <option value="rabbitmq">RabbitMQ</option>
                  <option value="sqs">AWS SQS</option>
                  <option value="pubsub">Google Pub/Sub</option>
                  <option value="kinesis">Kinesis</option>
                  <option value="other">Other</option>
                </select>
              </Field>

              <Field label="API Gateway">
                <select className="input-field" aria-label="API Gateway" value={apiGateway} onChange={(e) => setApiGateway(e.target.value)}>
                  <option value="">None / Direct</option>
                  <option value="aws_apigw">AWS API Gateway</option>
                  <option value="azure_apim">Azure APIM</option>
                  <option value="kong">Kong</option>
                  <option value="nginx">NGINX</option>
                  <option value="other">Other</option>
                </select>
              </Field>

              <Field label="Auth Mechanism">
                <select className="input-field" aria-label="Auth Mechanism" value={authMechanism} onChange={(e) => setAuthMechanism(e.target.value)}>
                  <option value="">Unknown</option>
                  <option value="oauth">OAuth 2.0</option>
                  <option value="api_key">API Key</option>
                  <option value="jwt">JWT</option>
                  <option value="saml">SAML</option>
                  <option value="other">Other</option>
                </select>
              </Field>

              <Field label="Monitoring / Observability">
                <select className="input-field" aria-label="Monitoring / Observability" value={monitoringTool} onChange={(e) => setMonitoringTool(e.target.value)}>
                  <option value="">None / Unknown</option>
                  <option value="datadog">Datadog</option>
                  <option value="cloudwatch">CloudWatch</option>
                  <option value="prometheus">Prometheus + Grafana</option>
                  <option value="grafana">Grafana</option>
                  <option value="newrelic">New Relic</option>
                  <option value="other">Other</option>
                </select>
              </Field>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
              {[
                { label: "Multi-region deployment", state: isMultiRegion,     set: setIsMultiRegion,     icon: "public" },
                { label: "Rate Limiting configured", state: hasRateLimiting,   set: setHasRateLimiting,   icon: "speed" },
                { label: "Retry logic with backoff", state: hasRetryLogic,     set: setHasRetryLogic,     icon: "replay" },
                { label: "Circuit breaker in place", state: hasCircuitBreaker, set: setHasCircuitBreaker, icon: "electric_bolt" },
                { label: "Response caching enabled", state: hasCaching,        set: setHasCaching,        icon: "cached" },
                { label: "Dead-letter queue (DLQ)",  state: hasDlq,            set: setHasDlq,            icon: "move_to_inbox" },
              ].map(({ label, state, set, icon }) => (
                <label key={label} className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-colors ${state ? "border-secondary/50 bg-secondary/5" : "border-[var(--color-outline-variant)] bg-[var(--color-surface-container-low)]"}`}>
                  <span className={`material-symbols-outlined text-base ${state ? "text-secondary" : "text-[var(--color-on-surface-variant)] opacity-40"}`}>{icon}</span>
                  <span className="text-xs font-semibold text-[var(--color-on-surface)] flex-1">{label}</span>
                  <div
                    onClick={() => set(!state)}
                    className={`w-9 h-5 rounded-full transition-colors flex-shrink-0 relative cursor-pointer ${state ? "bg-secondary" : "bg-[var(--color-outline-variant)]"}`}
                  >
                    <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${state ? "translate-x-4" : "translate-x-0.5"}`} />
                  </div>
                </label>
              ))}
            </div>

            <Field label="Additional Architecture Notes (optional)">
              <textarea
                className="input-field min-h-[72px] resize-y"
                placeholder="e.g. We use Kafka for message queuing, no response caching, single-region deployment on AWS us-east-1…"
                value={archNotes}
                onChange={(e) => setArchNotes(e.target.value)}
              />
              <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-50 mt-1">
                Free-form — mention any queues, caches, auth systems, or constraints. The RCA engine parses keywords automatically.
              </p>
            </Field>
          </>
        )}

        {/* ── Upload tab ──────────────────────────────────── */}
        {archTab === "upload" && (
          <div className="space-y-5">
            <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70">
              Upload an architecture document (PDF/TXT) or diagram image (PNG/JPEG/WEBP). The LLM will extract infrastructure details and auto-fill the form fields above.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Doc upload */}
              <div>
                <p className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-70 mb-2">Architecture Document</p>
                <div className="relative group p-5 border-2 border-dashed border-[var(--color-outline-variant)] rounded-2xl bg-[var(--color-surface-container-low)]/50 hover:border-primary/50 hover:bg-primary/5 transition-all text-center cursor-pointer">
                  <input
                    type="file"
                    accept=".txt,.pdf,.md"
                    aria-label="Upload Architecture Document"
                    className="absolute inset-0 opacity-0 cursor-pointer z-10"
                    onChange={(e) => setArchDocFile(e.target.files?.[0] || null)}
                  />
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2 ${archDocFile ? "bg-[#6bff8f] text-[#006e2f]" : "bg-primary/10 text-primary"}`}>
                    <span className="material-symbols-outlined text-xl">{archDocFile ? "task_alt" : "description"}</span>
                  </div>
                  {archDocFile ? (
                    <div>
                      <p className="font-bold text-sm truncate">{archDocFile.name}</p>
                      <p className="text-[10px] font-bold text-[#006e2f] uppercase tracking-widest mt-0.5">Ready</p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-sm font-semibold">Upload document</p>
                      <p className="text-[9px] font-bold uppercase tracking-widest opacity-50 mt-0.5">TXT · PDF · MD</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Diagram upload */}
              <div>
                <p className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-70 mb-2">Architecture Diagram</p>
                <div className="relative group p-5 border-2 border-dashed border-[var(--color-outline-variant)] rounded-2xl bg-[var(--color-surface-container-low)]/50 hover:border-primary/50 hover:bg-primary/5 transition-all text-center cursor-pointer">
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    aria-label="Upload Architecture Diagram"
                    className="absolute inset-0 opacity-0 cursor-pointer z-10"
                    onChange={(e) => setArchDiagramFile(e.target.files?.[0] || null)}
                  />
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2 ${archDiagramFile ? "bg-[#6bff8f] text-[#006e2f]" : "bg-primary/10 text-primary"}`}>
                    <span className="material-symbols-outlined text-xl">{archDiagramFile ? "task_alt" : "image"}</span>
                  </div>
                  {archDiagramFile ? (
                    <div>
                      <p className="font-bold text-sm truncate">{archDiagramFile.name}</p>
                      <p className="text-[10px] font-bold text-[#006e2f] uppercase tracking-widest mt-0.5">Ready</p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-sm font-semibold">Upload diagram</p>
                      <p className="text-[9px] font-bold uppercase tracking-widest opacity-50 mt-0.5">PNG · JPEG · WEBP (requires Azure OpenAI, Gemini, or OpenAI)</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={handleArchExtract}
              disabled={(!archDocFile && !archDiagramFile) || archExtracting}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-white text-sm font-bold hover:bg-primary/90 transition-colors disabled:opacity-40"
            >
              <span className={`material-symbols-outlined text-base ${archExtracting ? "animate-spin" : ""}`}>
                {archExtracting ? "progress_activity" : "auto_awesome"}
              </span>
              {archExtracting ? "Extracting…" : "Auto-Extract Architecture"}
            </button>

            <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-50">
              Uses your selected LLM provider to extract infrastructure details. Extracted fields will be auto-filled in the Structured Form tab.
            </p>
          </div>
        )}
      </Section>

      {/* ── Actions ────────────────────────────────────────── */}
      <div className="flex justify-end gap-4 pt-4">
        <button
          onClick={() => router.push("/dashboard")}
          className="px-6 py-3 rounded-xl border border-[var(--color-outline)] text-sm font-semibold hover:bg-[var(--color-surface-variant)] transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleNext}
          disabled={isNavigating}
          className="group flex items-center gap-2 px-8 py-3 rounded-xl btn-primary text-sm font-semibold disabled:opacity-50"
        >
          {isNavigating ? (
            <>
              <span className="material-symbols-outlined text-base animate-spin">progress_activity</span>
              Generating preview…
            </>
          ) : (
            <>
              Preview Tests
              <span className="material-symbols-outlined text-base group-hover:translate-x-1 transition-transform">arrow_forward</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}

// ── Small reusable components ──────────────────────────────────────────────

function Section({ title, icon, children }: { title: string; icon: string; children: React.ReactNode }) {
  return (
    <div className="card p-6 space-y-5">
      <div className="flex items-center gap-3">
        <span className="material-symbols-outlined text-primary">{icon}</span>
        <h2 className="font-headline text-lg font-black tracking-tight text-[var(--color-on-surface)]">{title}</h2>
      </div>
      {children}
    </div>
  );
}

function Field({ label, children, error }: { label: string; children: React.ReactNode; error?: string }) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-70">{label}</label>
      {children}
      {error && <p className="text-xs text-error">{error}</p>}
    </div>
  );
}

function SliderField({ label, min, max, value, onChange }: {
  label: string; min: number; max: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between">
        <label className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-70">{label}</label>
        <span className="text-xs font-black text-primary">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full accent-primary"
      />
      <div className="flex justify-between text-[10px] text-[var(--color-on-surface-variant)] opacity-40">
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  );
}

function CheckboxRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: () => void }) {
  return (
    <label className="flex items-center gap-3 cursor-pointer">
      <span
        onClick={onChange}
        className={`w-5 h-5 rounded flex items-center justify-center border transition-colors flex-shrink-0 ${
          checked ? "bg-primary border-primary" : "border-[var(--color-outline)] hover:border-primary"
        }`}
      >
        {checked && <span className="material-symbols-outlined text-white text-xs">check</span>}
      </span>
      <span className="text-sm text-[var(--color-on-surface)]">{label}</span>
    </label>
  );
}
