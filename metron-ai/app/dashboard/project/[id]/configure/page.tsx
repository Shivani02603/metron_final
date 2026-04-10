"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";

const API = "http://localhost:8000";

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
  const [connectionStatus, setConnectionStatus] = useState<"idle" | "testing" | "ok" | "fail">("idle");
  const [connectionMsg, setConnectionMsg] = useState("");

  // Agent
  const [agentName, setAgentName] = useState("");
  const [agentDomain, setAgentDomain] = useState("General");
  const [agentDescription, setAgentDescription] = useState("");

  // RAG
  const [isRag, setIsRag] = useState(false);
  const [ragText, setRagText] = useState("");

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
        if (data.documentText) setRagText(data.documentText);
        if (data.name) setAgentName(data.name);
      } catch (_) {}
    }
  }, [projectId]);

  // Load providers
  useEffect(() => {
    fetch(`${API}/api/providers`)
      .then((r) => r.json())
      .then((data) => setProviders(data))
      .catch(() => {});
  }, []);

  const testConnection = async () => {
    if (!endpointUrl) return;
    setConnectionStatus("testing");
    try {
      const res = await fetch(`${API}/api/connect-test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint_url: endpointUrl,
          request_field: requestField,
          response_field: responseField,
          auth_type: authType,
          auth_token: authToken,
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
      const res = await fetch(`${API}/api/preview`, {
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

    // Build full config and store
    const fullConfig = {
      endpoint_url: endpointUrl,
      request_field: requestField,
      response_field: responseField,
      auth_type: authType,
      auth_token: authToken,
      agent_name: agentName,
      agent_domain: agentDomain,
      agent_description: agentDescription,
      is_rag: isRag,
      rag_text: isRag ? ragText : "",
      num_personas: numPersonas,
      num_scenarios: numScenarios,
      conversation_turns: convTurns,
      enable_judge: enableJudge,
      performance_requests: perfRequests,
      load_concurrent_users: loadUsers,
      load_duration_seconds: loadDuration,
      llm_provider: llmProvider,
      llm_api_key: llmApiKey,
      selected_attacks: selectedAttacks,
      attacks_per_category: attacksPerCategory,
      ragas_metrics: ragasMetrics,
      deepeval_metrics: deepevalMetrics,
      use_geval: useGeval,
      application_type: applicationType,
    };

    sessionStorage.setItem(`fullconfig_${projectId}`, JSON.stringify(fullConfig));

    // Generate personas/scenarios only if not already done
    if (personas.length === 0) {
      try {
        const res = await fetch(`${API}/api/preview`, {
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
            </Field>
            <Field label="Response Field">
              <input className="input-field" placeholder="response" value={responseField} onChange={(e) => setResponseField(e.target.value)} />
            </Field>
          </div>
          <div className="space-y-4">
            <Field label="Authentication">
              <select className="input-field" value={authType} onChange={(e) => setAuthType(e.target.value as "none" | "bearer")}>
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
              <select className="input-field" value={agentDomain} onChange={(e) => setAgentDomain(e.target.value)}>
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
          <Field label="Ground Truth / Knowledge Base">
            <textarea
              className="input-field resize-none h-[140px] font-mono text-xs"
              placeholder="Paste your ground truth documents or knowledge base content here…"
              value={ragText}
              onChange={(e) => setRagText(e.target.value)}
            />
          </Field>
        )}
      </Section>

      {/* ── Test Parameters ────────────────────────────────── */}
      <Section title="Test Parameters" icon="tune">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-4">
            <SliderField label="Personas" min={1} max={20} value={numPersonas} onChange={setNumPersonas} />
            <SliderField label="Scenarios" min={1} max={30} value={numScenarios} onChange={setNumScenarios} />
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
            <SliderField label="Performance Requests" min={5} max={100} value={perfRequests} onChange={setPerfRequests} />
            <SliderField label="Load Test Users" min={1} max={50} value={loadUsers} onChange={setLoadUsers} />
            <SliderField label="Load Duration (s)" min={10} max={120} value={loadDuration} onChange={setLoadDuration} />
          </div>
        </div>
      </Section>

      {/* ── LLM Provider ──────────────────────────────────── */}
      <Section title="LLM Provider" icon="psychology">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-4">
            <Field label="Provider">
              <select className="input-field" value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)}>
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
            {providerInfo && (
              <div className="p-3 rounded-xl bg-[var(--color-surface-variant)] space-y-1">
                <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">Default model</p>
                <p className="text-xs font-mono font-semibold text-[var(--color-on-surface)]">{providerInfo.default}</p>
              </div>
            )}
          </div>
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
        <SliderField label="Attack Prompts per Category" min={1} max={10} value={attacksPerCategory} onChange={setAttacksPerCategory} />
      </Section>

      {/* ── Quality Metrics ────────────────────────────────── */}
      <Section title="Quality Metrics" icon="grade">
        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70 -mt-2">These metrics always run on every conversation.</p>
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
            <p className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">RAG Mode — also runs</p>
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
