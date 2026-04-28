"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";

const API = "";

interface Persona {
  id: string;
  name: string;
  description: string;
  traits: string[];
  sample_prompts: string[];
  intent?: string;
  fishbone?: Record<string, string>;
}

interface Scenario {
  id: string;
  name: string;
  description: string;
  initial_prompt: string;
  expected_behavior: string;
  category: string;
}

interface ToolStatus {
  installed: boolean;
  description: string;
  used_for?: string;
}

export default function PreviewPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [fullConfig, setFullConfig] = useState<Record<string, unknown> | null>(null);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [toolStatus, setToolStatus] = useState<Record<string, ToolStatus>>({});
  const [expandedPersonas, setExpandedPersonas] = useState<Set<string>>(new Set());
  const [expandedScenarios, setExpandedScenarios] = useState<Set<string>>(new Set());
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    // Load config
    const cfgStr = sessionStorage.getItem(`fullconfig_${projectId}`);
    if (cfgStr) setFullConfig(JSON.parse(cfgStr));

    // Load preview (personas/scenarios generated during configure)
    const previewStr = sessionStorage.getItem(`preview_${projectId}`);
    if (previewStr) {
      const preview = JSON.parse(previewStr);
      setPersonas(preview.personas || []);
      setScenarios(preview.scenarios || []);
    }

    // Fetch tool status
    fetch(`${API}/api/tools/status`)
      .then((r) => r.json())
      .then(setToolStatus)
      .catch(() => {});
  }, [projectId]);

  const togglePersona = (id: string) => {
    setExpandedPersonas((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleScenario = (id: string) => {
    setExpandedScenarios((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const handleRun = async () => {
    if (!fullConfig) return;
    setIsRunning(true);
    setError("");
    try {
      const cfg = fullConfig as Record<string, unknown>;
      const formData = new FormData();
      formData.append("config", JSON.stringify({
        endpoint_url: cfg.endpoint_url,
        request_field: cfg.request_field,
        response_field: cfg.response_field,
        auth_type: cfg.auth_type,
        auth_token: cfg.auth_token,
        request_template: cfg.request_template || null,
        response_trim_marker: cfg.response_trim_marker || null,
        agent_name: cfg.agent_name,
        agent_domain: cfg.agent_domain,
        agent_description: cfg.agent_description,
        is_rag: cfg.is_rag,
        num_personas: cfg.num_personas,
        num_scenarios: cfg.num_scenarios,
        conversation_turns: cfg.conversation_turns,
        enable_judge: cfg.enable_judge,
        performance_requests: cfg.performance_requests,
        load_concurrent_users: cfg.load_concurrent_users,
        load_duration_seconds: cfg.load_duration_seconds,
        llm_provider: cfg.llm_provider,
        llm_api_key: cfg.llm_api_key,
        application_type: cfg.application_type || "chatbot",
        selected_attacks: cfg.selected_attacks,
        attacks_per_category: cfg.attacks_per_category,
        deployment_type: cfg.deployment_type,
        additional_architecture_notes: cfg.additional_architecture_notes,
      }));

      // Attach RAG knowledge base document if present
      const ragText = cfg.rag_text as string;
      if (ragText) {
        const blob = new Blob([ragText], { type: "text/plain" });
        formData.append("document", new File([blob], "knowledge.txt"));
      }

      // Attach ground truth file if present (RAG mode)
      const gtText = sessionStorage.getItem(`ground_truth_${projectId}`);
      const gtName = sessionStorage.getItem(`ground_truth_name_${projectId}`) || "ground_truth.csv";
      if (gtText) {
        const gtBlob = new Blob([gtText], { type: "text/plain" });
        formData.append("ground_truth_file", new File([gtBlob], gtName));
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout for backend response

      try {
        const res = await fetch(`${API}/api/run`, { method: "POST", body: formData, signal: controller.signal });
        clearTimeout(timeoutId);
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error((err as { detail: string }).detail || "Failed to start run");
        }
        const { run_id } = await res.json();

        // Store mapping: projectId → run_id
        sessionStorage.setItem(`run_id_${projectId}`, run_id);
        router.push(`/dashboard/project/${projectId}/run`);
      } catch (fetchError) {
        clearTimeout(timeoutId);
        if (fetchError instanceof Error && fetchError.name === 'AbortError') {
          throw new Error("Request timeout. Backend API is not responding.");
        }
        throw fetchError;
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setIsRunning(false);
    }
  };

  const cfg = fullConfig as Record<string, unknown> | null;

  // Calculate test counts
  const numPersonas = personas.length || Number(cfg?.num_personas) || 3;
  // functional: 3 prompts per persona × 3 DeepEval metrics + GEval criteria per conversation
  const funcCount = numPersonas * 3 * 3;
  // security: attack prompts + golden dataset (20) across all conversations
  const attacksPerCat = Number(cfg?.attacks_per_category) || 3;
  const secCount = 5 * attacksPerCat + 20;
  // quality: GEval criteria per persona conversation (~4 criteria + overall)
  const qualCount = numPersonas * 3 * (cfg?.is_rag ? 8 : 5);
  const perfCount = Number(cfg?.performance_requests) || 20;
  const loadCount = Number(cfg?.load_concurrent_users) || 5;

  // Icon mapping for known tools
  const TOOL_ICONS: Record<string, string> = {
    presidio: "policy",
    detoxify: "block",
    llm_guard: "shield",
    deepeval: "grade",
    ragas: "assessment",
    rouge_score: "text_compare",
    bert_score: "psychology",
    garak: "security",
    neo4j: "hub",
  };

  return (
    <div className="max-w-4xl mx-auto pb-20 space-y-10 animate-fade-in">
      {/* Header */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-primary" />
          <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">Step 2 of 3</span>
        </div>
        <h1 className="font-headline text-4xl font-black text-[var(--color-on-surface)] tracking-tighter">Preview Test Plan</h1>
        <p className="text-[var(--color-on-surface-variant)] text-sm font-medium opacity-60">Review what will be tested before launching.</p>
      </div>

      {/* Config Summary */}
      {cfg && (
        <div className="card p-6">
          <h2 className="font-headline text-base font-black tracking-tight text-[var(--color-on-surface)] mb-4">Configuration Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
            <div className="space-y-2">
              <SummaryRow label="Endpoint" value={cfg.endpoint_url as string} mono />
              <SummaryRow label="Auth" value={cfg.auth_type === "bearer" ? "Bearer Token" : "None"} />
              <SummaryRow label="Request Field" value={cfg.request_field as string} mono />
              <SummaryRow label="Response Field" value={cfg.response_field as string} mono />
            </div>
            <div className="space-y-2">
              <SummaryRow label="Agent" value={(cfg.agent_name as string) || "—"} />
              <SummaryRow label="Domain" value={cfg.agent_domain as string} />
              <SummaryRow label="RAG Mode" value={cfg.is_rag ? "Yes" : "No"} />
              <SummaryRow label="LLM Provider" value={cfg.llm_provider as string} />
            </div>
          </div>
        </div>
      )}

      {/* Tool Status */}
      <div className="card p-6">
        <h2 className="font-headline text-base font-black tracking-tight text-[var(--color-on-surface)] mb-4">
          Testing Tools
          {Object.keys(toolStatus).length > 0 && (
            <span className="ml-2 text-xs font-normal text-[var(--color-on-surface-variant)] opacity-60">
              {Object.values(toolStatus).filter((t) => t.installed).length}/{Object.keys(toolStatus).length} installed
            </span>
          )}
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {Object.entries(toolStatus).map(([key, status]) => {
            const installed = status.installed;
            const icon = TOOL_ICONS[key] || "build";
            const label = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
            return (
              <div key={key} className={`p-3 rounded-xl border ${installed ? "border-[var(--color-secondary)] bg-[rgba(0,110,47,0.04)]" : "border-[var(--color-outline-variant)]"}`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className={`material-symbols-outlined text-sm ${installed ? "text-secondary" : "text-[var(--color-on-surface-variant)]"}`}>{icon}</span>
                  <span className="text-xs font-black">{label}</span>
                  <span className={`ml-auto text-[10px] font-bold ${installed ? "text-secondary" : "text-[var(--color-on-surface-variant)] opacity-60"}`}>
                    {installed ? "✓" : "⚠"}
                  </span>
                </div>
                <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60 leading-tight">{status.used_for || status.description}</p>
              </div>
            );
          })}
        </div>
        {Object.keys(toolStatus).length === 0 && (
          <p className="text-sm text-[var(--color-on-surface-variant)] opacity-60">Loading tool status…</p>
        )}
      </div>

      {/* Test Count Summary */}
      <div className="card p-6">
        <h2 className="font-headline text-base font-black tracking-tight text-[var(--color-on-surface)] mb-4">Test Counts</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: "Functional", count: funcCount, icon: "science", color: "text-primary" },
            { label: "Security", count: secCount, icon: "security", color: "text-error" },
            { label: "Quality", count: qualCount, icon: "grade", color: "text-secondary" },
            { label: "Performance", count: `${perfCount} req`, icon: "speed", color: "text-[var(--color-tertiary,#855300)]" },
            { label: "Load Test", count: `${loadCount} users`, icon: "group", color: "text-[var(--color-on-surface-variant)]" },
          ].map((item) => (
            <div key={item.label} className="text-center p-3 rounded-xl bg-[var(--color-surface-container-low)]">
              <span className={`material-symbols-outlined text-2xl ${item.color}`}>{item.icon}</span>
              <p className="font-headline text-2xl font-black text-[var(--color-on-surface)] mt-1">{item.count}</p>
              <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60 mt-0.5">{item.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Generated Personas */}
      {personas.length > 0 && (
        <div className="card p-6">
          <h2 className="font-headline text-base font-black tracking-tight text-[var(--color-on-surface)] mb-4">
            Generated Personas
            <span className="ml-2 text-xs font-normal text-[var(--color-on-surface-variant)] opacity-60">({personas.length})</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {personas.map((p) => (
              <div key={p.id} className="p-4 rounded-xl border border-[var(--color-outline-variant)]">
                <button
                  onClick={() => togglePersona(p.id)}
                  className="w-full text-left"
                >
                  <div className="flex items-center justify-between">
                    <p className="font-headline text-sm font-black">{p.name}</p>
                    <span className="material-symbols-outlined text-base text-[var(--color-on-surface-variant)]">
                      {expandedPersonas.has(p.id) ? "expand_less" : "expand_more"}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60 mt-1 line-clamp-2">{p.description}</p>
                  {(p.intent || p.fishbone) && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {(p.intent || p.fishbone?.intent) && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${
                          (p.intent || p.fishbone?.intent) === "adversarial" ? "bg-error/10 text-error" :
                          (p.intent || p.fishbone?.intent) === "edge_case" ? "bg-[#855300]/10 text-[#855300]" :
                          "bg-primary/10 text-primary"
                        }`}>{p.intent || p.fishbone?.intent}</span>
                      )}
                      {p.fishbone?.expertise && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)] font-semibold">{p.fishbone.expertise}</span>
                      )}
                      {p.fishbone?.emotional_state && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)] font-semibold">{p.fishbone.emotional_state}</span>
                      )}
                    </div>
                  )}
                </button>
                {expandedPersonas.has(p.id) && (
                  <div className="mt-3 space-y-2 border-t border-[var(--color-outline-variant)] pt-3">
                    {p.traits.length > 0 && (
                      <div>
                        <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-on-surface-variant)] opacity-60 mb-1">Traits</p>
                        <div className="flex flex-wrap gap-1">
                          {p.traits.map((t) => (
                            <span key={t} className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">{t}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {p.sample_prompts.length > 0 && (
                      <div>
                        <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-on-surface-variant)] opacity-60 mb-1">Sample Prompts</p>
                        <ul className="space-y-1">
                          {p.sample_prompts.slice(0, 3).map((prompt, j) => (
                            <li key={j} className="text-xs text-[var(--color-on-surface)] pl-3 border-l-2 border-primary/30">"{prompt}"</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Generated Scenarios */}
      {scenarios.length > 0 && (
        <div className="card p-6">
          <h2 className="font-headline text-base font-black tracking-tight text-[var(--color-on-surface)] mb-4">
            Test Scenarios
            <span className="ml-2 text-xs font-normal text-[var(--color-on-surface-variant)] opacity-60">({scenarios.length})</span>
          </h2>
          <div className="space-y-2">
            {scenarios.map((s) => (
              <div key={s.id} className="border border-[var(--color-outline-variant)] rounded-xl overflow-hidden">
                <button
                  onClick={() => toggleScenario(s.id)}
                  className="w-full flex items-center justify-between p-4 text-left hover:bg-[var(--color-surface-container-low)] transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="material-symbols-outlined text-base text-primary">assignment</span>
                    <div>
                      <p className="text-sm font-black">{s.name}</p>
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--color-surface-container-low)] text-[var(--color-on-surface-variant)] border border-[var(--color-outline-variant)]">{s.category}</span>
                    </div>
                  </div>
                  <span className="material-symbols-outlined text-base text-[var(--color-on-surface-variant)]">
                    {expandedScenarios.has(s.id) ? "expand_less" : "expand_more"}
                  </span>
                </button>
                {expandedScenarios.has(s.id) && (
                  <div className="px-4 pb-4 space-y-2 border-t border-[var(--color-outline-variant)]">
                    <p className="text-xs text-[var(--color-on-surface-variant)] mt-3">{s.description}</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="p-3 rounded-lg bg-[var(--color-surface-container-low)]">
                        <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-on-surface-variant)] opacity-60 mb-1">Initial Prompt</p>
                        <p className="text-xs font-mono">{s.initial_prompt}</p>
                      </div>
                      <div className="p-3 rounded-lg bg-[var(--color-surface-container-low)]">
                        <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-on-surface-variant)] opacity-60 mb-1">Expected Behavior</p>
                        <p className="text-xs">{s.expected_behavior}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      {error && (
        <div className="p-4 rounded-xl bg-error/10 border border-error/20 flex items-center gap-3">
          <span className="material-symbols-outlined text-error">error</span>
          <p className="text-sm text-error">{error}</p>
        </div>
      )}

      <div className="flex justify-between gap-4 pt-4">
        <button
          onClick={() => router.push(`/dashboard/project/${projectId}/configure`)}
          className="flex items-center gap-2 px-6 py-3 rounded-xl border border-[var(--color-outline)] text-sm font-semibold hover:bg-[var(--color-surface-variant)] transition-colors"
        >
          <span className="material-symbols-outlined text-base">arrow_back</span>
          Back to Config
        </button>
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="group flex items-center gap-2 px-8 py-3 rounded-xl btn-primary text-sm font-semibold disabled:opacity-50"
        >
          {isRunning ? (
            <>
              <span className="material-symbols-outlined text-base animate-spin">progress_activity</span>
              Starting…
            </>
          ) : (
            <>
              <span className="material-symbols-outlined text-base">rocket_launch</span>
              Run All Tests
            </>
          )}
        </button>
      </div>
    </div>
  );
}

function SummaryRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-start gap-3">
      <span className="text-xs text-[var(--color-on-surface-variant)] opacity-60 w-24 flex-shrink-0">{label}</span>
      <span className={`text-xs font-semibold text-[var(--color-on-surface)] break-all ${mono ? "font-mono" : ""}`}>{value || "—"}</span>
    </div>
  );
}
