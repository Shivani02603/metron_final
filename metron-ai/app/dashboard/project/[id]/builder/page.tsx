"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { authFetch } from "@/lib/api";

interface TestModule {
  id: string;
  name: string;
  icon: string;
  description: string;
  enabled: boolean;
  color: string;
  type: string;
  testClass: string;
}

interface ProjectConfig {
  name: string;
  endpoint: string;
  apiKey: string;
  documentText: string;
  documentName: string;
}

const MOD_BORDER_L: Record<string, string> = {
  "#00668a": "[border-left-color:#00668a]",
  "#ba1a1a": "[border-left-color:#ba1a1a]",
  "#38bdf8": "[border-left-color:#38bdf8]",
  "#006e2f": "[border-left-color:#006e2f]",
};
const MOD_TEXT: Record<string, string> = {
  "#00668a": "text-[#00668a]",
  "#ba1a1a": "text-[#ba1a1a]",
  "#38bdf8": "text-[#38bdf8]",
  "#006e2f": "text-[#006e2f]",
};

export default function NodeBuilder() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const [error, setError] = useState("");

  // Advanced field config
  const [inputField, setInputField] = useState("");
  const [outputField, setOutputField] = useState("");
  const [loadUsers, setLoadUsers] = useState(50);
  const [loadDuration, setLoadDuration] = useState(60);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [modules, setModules] = useState<TestModule[]>([
    { id: "func", name: "Functional Accuracy", icon: "fact_check", description: "Answer relevancy, faithfulness, usefulness, hallucination, ROUGE, BERTScore.", enabled: true, color: "#00668a", type: "Core", testClass: "functional" },
    { id: "sec", name: "Security Audit", icon: "shield", description: "Jailbreak, prompt injection, PII leakage, bias, toxicity.", enabled: false, color: "#ba1a1a", type: "Safety", testClass: "security" },
    { id: "perf", name: "Performance", icon: "bolt", description: "Latency p50/p95/p99, moderation scoring.", enabled: false, color: "#38bdf8", type: "System", testClass: "performance" },
    { id: "load", name: "Load Stability", icon: "stacked_line_chart", description: "Locust concurrent user stress test (takes longer).", enabled: false, color: "#006e2f", type: "Scale", testClass: "load" },
  ]);

  useEffect(() => {
    const stored = sessionStorage.getItem(`project_${projectId}`);
    if (stored) setConfig(JSON.parse(stored));
  }, [projectId]);

  const toggleModule = (id: string) => {
    setModules(modules.map((m) => (m.id === id ? { ...m, enabled: !m.enabled } : m)));
  };

  const activeModules = modules.filter((m) => m.enabled);

  const launchRun = async () => {
    if (!config || activeModules.length === 0) return;
    setIsLaunching(true);
    setError("");
    try {
      const blob = new Blob([config.documentText], { type: "text/plain" });
      const file = new File([blob], config.documentName || "document.txt");
      const formData = new FormData();
      formData.append("document", file);

      const testClasses = activeModules.map((m) => m.testClass);
      
      formData.append(
        "config",
        JSON.stringify({
          project_id: projectId,
          endpoint_url: config.endpoint,
          request_field: inputField || "message",
          response_field: outputField || "response",
          auth_token: config.apiKey,
          auth_type: config.apiKey ? "bearer" : "none",
          agent_name: config.name || "Default Agent",
          application_type: "chatbot",
          is_rag: false,
          llm_provider: "Groq", // default or fetch from project later
          llm_api_key: "", // will be injected on backend via env if blank
          num_personas: 3,
          num_scenarios: 5,
          conversation_turns: 3,
          enable_judge: true,
          performance_requests: testClasses.includes("performance") ? loadUsers : 0,
          load_concurrent_users: loadUsers,
          load_duration_seconds: loadDuration,
        })
      );

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout for backend response

      try {
        const res = await authFetch("/api/run", {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || "Run failed");
        }

        const { run_id } = await res.json();
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
      setIsLaunching(false);
    }
  };

  return (
    <div className="h-[calc(100vh-160px)] flex flex-col -m-8 relative overflow-hidden bg-[var(--color-surface-container-low)]">

      {/* ─── Control Header ───────────────────────────── */}
      <div className="h-16 bg-white border-b border-outline-variant/10 flex items-center justify-between px-8 relative z-50">
        <div className="flex items-center gap-4">
          <button onClick={() => router.back()} className="text-outline hover:text-primary transition-colors flex items-center gap-2">
            <span className="material-symbols-outlined text-lg">arrow_back</span>
            <span className="text-[10px] font-black uppercase tracking-widest leading-none">Strategy Hub</span>
          </button>
          <div className="w-px h-6 bg-outline-variant/20 mx-2" />
          <h1 className="font-headline font-black text-xl text-on-surface tracking-tighter">Node Designer</h1>
          {config && (
            <span className="text-[9px] font-bold text-outline opacity-40 truncate max-w-[200px]">{config.name}</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {error && <span className="text-[9px] text-red-500 font-bold max-w-[200px] truncate">{error}</span>}
          <div className="px-3 py-1.5 rounded-full bg-primary/5 text-[9px] font-black text-primary border border-primary/10 uppercase tracking-[.2em]">Blueprint Active</div>
          <button
            type="button"
            onClick={launchRun}
            disabled={isLaunching || activeModules.length === 0}
            className="px-6 py-2.5 rounded-xl btn-primary text-[10px] font-black uppercase tracking-widest shadow-xl shadow-primary/10 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLaunching ? (
              <>
                <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
                Starting...
              </>
            ) : (
              <>
                Run Configuration
                <span className="material-symbols-outlined text-sm">rocket_launch</span>
              </>
            )}
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden relative">

        {/* ─── Left Sidebar ───────────────────────────────── */}
        <aside className="w-80 bg-white border-r border-outline-variant/10 p-6 flex flex-col gap-6 overflow-y-auto no-scrollbar relative z-20">
          <div className="space-y-1 pb-4 border-b border-outline-variant/10">
            <h3 className="font-headline font-black text-[10px] uppercase tracking-[0.2em] text-outline">Compute Modules</h3>
            <p className="text-[9px] font-medium text-outline opacity-60 italic">Define your intelligence pipeline.</p>
          </div>

          <div className="space-y-3">
            {modules.map((mod) => (
              <div
                key={mod.id}
                onClick={() => toggleModule(mod.id)}
                className={`p-4 rounded-3xl border transition-all cursor-pointer group relative overflow-hidden ${
                  mod.enabled
                    ? "bg-primary/5 border-primary/20 ring-1 ring-primary/10"
                    : "bg-transparent border-outline-variant/10 grayscale opacity-40 hover:opacity-100 hover:grayscale-0"
                }`}
              >
                {mod.enabled && <div className="absolute top-0 right-0 w-12 h-12 bg-primary/5 rounded-full -mr-6 -mt-6 blur-xl" />}
                <div className="flex items-center justify-between mb-3 relative z-10">
                  <div className="w-10 h-10 rounded-2xl flex items-center justify-center bg-white border border-outline-variant/10 shadow-sm group-hover:scale-110 transition-transform">
                    <span className={`material-symbols-outlined text-xl ${mod.enabled ? MOD_TEXT[mod.color] || "" : "text-[var(--color-outline)]"}`}>{mod.icon}</span>
                  </div>
                  <div className={`w-8 h-4 rounded-full relative transition-colors duration-300 ${mod.enabled ? "bg-primary" : "bg-outline-variant/30"}`}>
                    <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm transition-all duration-300 ${mod.enabled ? "left-[18px]" : "left-0.5"}`} />
                  </div>
                </div>
                <h4 className="font-headline font-black text-xs text-on-surface mb-1 flex items-center gap-2">
                  {mod.name}
                  <span className="text-[8px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded bg-outline-variant/10 text-outline opacity-40">{mod.type}</span>
                </h4>
                <p className="text-[9px] font-medium text-outline leading-normal opacity-70">{mod.description}</p>
              </div>
            ))}
          </div>

          {/* Advanced Config */}
          <div className="border-t border-outline-variant/10 pt-4">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-[9px] font-black uppercase tracking-widest text-outline opacity-60 hover:opacity-100 hover:text-primary transition-all w-full"
            >
              <span className="material-symbols-outlined text-sm">{showAdvanced ? "expand_less" : "expand_more"}</span>
              Advanced Config
            </button>

            {showAdvanced && (
              <div className="mt-4 space-y-3">
                <div className="space-y-1">
                  <label className="text-[8px] font-black uppercase tracking-widest text-outline opacity-50">Input field name</label>
                  <input
                    type="text"
                    placeholder="message"
                    value={inputField}
                    onChange={(e) => setInputField(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-primary outline-none text-xs font-bold text-on-surface"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[8px] font-black uppercase tracking-widest text-outline opacity-50">Output field name</label>
                  <input
                    type="text"
                    placeholder="response"
                    value={outputField}
                    onChange={(e) => setOutputField(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-primary outline-none text-xs font-bold text-on-surface"
                  />
                </div>
                {modules.find((m) => m.id === "load" && m.enabled) && (
                  <>
                    <div className="space-y-1">
                      <label className="text-[8px] font-black uppercase tracking-widest text-outline opacity-50">Concurrent users</label>
                      <input
                        type="number"
                        min={1}
                        max={500}
                        value={loadUsers}
                        onChange={(e) => setLoadUsers(parseInt(e.target.value) || 50)}
                        title="Concurrent users"
                        aria-label="Concurrent users"
                        className="w-full px-3 py-2 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-primary outline-none text-xs font-bold text-on-surface"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-[8px] font-black uppercase tracking-widest text-outline opacity-50">Duration (seconds)</label>
                      <input
                        type="number"
                        min={10}
                        max={600}
                        value={loadDuration}
                        onChange={(e) => setLoadDuration(parseInt(e.target.value) || 60)}
                        title="Duration in seconds"
                        aria-label="Duration in seconds"
                        className="w-full px-3 py-2 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-primary outline-none text-xs font-bold text-on-surface"
                      />
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </aside>

        {/* ─── Main Canvas ──────────────────────────────────── */}
        <div className="flex-1 relative bg-[radial-gradient(circle_at_center,_#001e2c10_1.2px,_transparent_1.2px)] bg-[size:40px_40px]">
          <div className="absolute inset-0 flex items-center justify-center animate-fade-in p-20 overflow-x-auto no-scrollbar">
            <div className="flex items-center gap-0 relative">

              {/* 1. Input */}
              <div className="flex flex-col items-center gap-4 relative z-10">
                <div className="w-20 h-20 rounded-3xl bg-white border-2 border-primary/20 shadow-xl flex items-center justify-center relative hover:scale-110 transition-transform cursor-help group">
                  <span className="material-symbols-outlined text-primary text-3xl">description</span>
                  <div className="absolute -bottom-2 -right-2 w-7 h-7 rounded-full bg-[#6bff8f] border-4 border-white flex items-center justify-center text-on-surface shadow-lg">
                    <span className="material-symbols-outlined text-xs font-black">check</span>
                  </div>
                </div>
                <p className="text-[9px] font-black uppercase tracking-[0.2em] text-outline opacity-50">Knowledge Seed</p>
              </div>

              {/* Arrow */}
              <div className="w-16 h-[2px] bg-primary/20 relative">
                <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-2 h-2 rounded-full bg-primary/40" />
              </div>

              {/* 2. Engine */}
              <div className="flex flex-col items-center gap-4 relative z-10">
                <div className="w-28 h-28 rounded-[2.5rem] bg-gradient-to-br from-[#001e2c] to-[#01405e] border-4 border-white/10 shadow-[0_32px_64px_-12px_rgba(0,102,138,0.4)] flex flex-col items-center justify-center text-white p-4">
                  <span className="material-symbols-outlined text-3xl mb-1 text-primary-container animate-pulse">hub</span>
                  <p className="text-[8px] font-black uppercase tracking-widest opacity-40 leading-none mb-1">Target Engine</p>
                  <p className="text-[10px] font-black text-primary-container uppercase truncate w-full text-center tracking-tighter">Metron_Core</p>
                </div>
                <p className="text-[9px] font-black uppercase tracking-[0.2em] text-on-surface">Intelligence Hub</p>
              </div>

              {/* 3. Branches */}
              <div className="flex flex-col items-center gap-6 relative ml-16">
                {activeModules.length === 0 ? (
                  <div className="w-64 py-12 border-2 border-dashed border-outline-variant/20 rounded-[2.5rem] flex flex-col items-center justify-center text-center px-8">
                    <span className="material-symbols-outlined text-outline opacity-20 text-3xl mb-2">schema</span>
                    <p className="text-[9px] font-black uppercase tracking-widest text-outline opacity-30">Enable modules to build pipeline</p>
                  </div>
                ) : (
                  <div className="flex flex-col gap-5 animate-fade-in relative pl-8">
                    <div className="absolute -left-16 top-1/2 -translate-y-1/2 w-16 h-[2px] bg-primary/20" />
                    <div className="absolute -left-[1px] top-4 bottom-4 w-[2px] bg-primary/20" />
                    {activeModules.map((mod) => (
                      <div key={mod.id} className="flex items-center relative group">
                        <div className="w-8 h-[2px] bg-primary/20 group-hover:bg-primary transition-colors" />
                        <div className={`px-5 py-3.5 rounded-2xl bg-white border border-outline-variant/20 shadow-md flex items-center gap-3 hover:shadow-xl hover:-translate-y-0.5 transition-all w-56 border-l-4 ${MOD_BORDER_L[mod.color] || ""}`}>
                          <div className={`w-9 h-9 rounded-xl flex items-center justify-center bg-gray-50/50 shadow-inner ${MOD_TEXT[mod.color] || ""}`}>
                            <span className="material-symbols-outlined text-xl">{mod.icon}</span>
                          </div>
                          <div className="flex-1 overflow-hidden">
                            <h5 className="font-headline font-black text-[11px] text-on-surface truncate">{mod.name}</h5>
                            <p className="text-[8px] font-bold uppercase tracking-widest text-outline/40 truncate">Ready</p>
                          </div>
                        </div>
                        <div className="w-12 h-[2px] bg-primary/10 group-hover:bg-primary/30 transition-colors" />
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* 4. Output */}
              {activeModules.length > 0 && (
                <div className="flex items-center animate-fade-in relative">
                  <div className="absolute -left-1 top-4 bottom-4 w-[2px] bg-primary/10" />
                  <div className="w-12 h-[2px] bg-primary/20 relative">
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-4 h-4 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
                      <span className="material-symbols-outlined text-[10px] font-black">chevron_right</span>
                    </div>
                  </div>
                  <div className="flex flex-col items-center gap-4 relative z-10 ml-8">
                    <div className="w-32 h-32 rounded-[2.5rem] bg-white border-4 border-primary/10 shadow-2xl flex flex-col items-center justify-center p-6 text-center group hover:scale-105 transition-all">
                      <div className="w-14 h-14 rounded-2xl bg-primary/5 flex items-center justify-center text-primary mb-2 group-hover:bg-primary group-hover:text-white transition-all">
                        <span className="material-symbols-outlined text-3xl">analytics</span>
                      </div>
                      <p className="text-[10px] font-black text-on-surface uppercase tracking-widest leading-none">Export</p>
                      <p className="text-[8px] font-bold text-outline uppercase tracking-widest opacity-60 mt-1">Intelligence</p>
                    </div>
                    <p className="text-[9px] font-black uppercase tracking-[0.2em] text-on-surface">Final Analysis</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Canvas overlays */}
          <div className="absolute top-8 right-8 flex flex-col gap-3">
            <div className="px-4 py-2 rounded-xl bg-white/50 backdrop-blur-md border border-outline-variant/10 text-[9px] font-bold text-outline/60 uppercase tracking-widest">
              Live Blueprint Canvas
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
