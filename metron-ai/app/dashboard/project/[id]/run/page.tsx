"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";

const API = "http://localhost:8000";
const POLL_INTERVAL = 3000;

interface PhaseResult {
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  avg_score: number;
  results?: Array<{
    test_name: string;
    passed: boolean;
    score: number;
    category: string;
  }>;
}

interface JobStatus {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  current_phase: string | null;
  phase_results: Record<string, PhaseResult | Record<string, number>>;
  error?: string;
}

const PHASES = [
  { id: "functional",  label: "Functional Tests",   icon: "science",      color: "text-primary",                          bgColor: "bg-primary/10",                          start: 0,  end: 35 },
  { id: "security",   label: "Security Tests",      icon: "security",     color: "text-error",                            bgColor: "bg-error/10",                            start: 35, end: 55 },
  { id: "quality",    label: "Quality Metrics",     icon: "grade",        color: "text-secondary",                        bgColor: "bg-secondary/10",                        start: 55, end: 70 },
  { id: "performance",label: "Performance Tests",   icon: "speed",        color: "text-[#855300]",                        bgColor: "bg-[#855300]/10",                        start: 70, end: 80 },
  { id: "load",       label: "Load Tests",          icon: "group",        color: "text-[var(--color-on-surface-variant)]",bgColor: "bg-[var(--color-surface-variant)]",      start: 80, end: 90 },
  { id: "feedback",   label: "Adaptive Feedback",   icon: "psychology",   color: "text-[var(--color-tertiary)]",          bgColor: "bg-[var(--color-tertiary-container)]/30",start: 90, end: 97 },
  { id: "report",     label: "Generating Report",   icon: "description",  color: "text-primary",                          bgColor: "bg-primary/10",                          start: 97, end: 100 },
];

export default function RunPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [runId, setRunId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [completedPhases, setCompletedPhases] = useState<Set<string>>(new Set());
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load run_id from sessionStorage
  useEffect(() => {
    const id = sessionStorage.getItem(`run_id_${projectId}`);
    setRunId(id);
  }, [projectId]);

  const poll = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API}/api/job/${id}/status`);
      if (!res.ok) return;
      const data: JobStatus = await res.json();
      setJobStatus(data);

      // Track completed phases
      if (data.phase_results) {
        setCompletedPhases(new Set(Object.keys(data.phase_results)));
      }

      if (data.status === "completed" || data.status === "failed") {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch (_) {}
  }, []);

  useEffect(() => {
    if (!runId) return;
    poll(runId);
    pollRef.current = setInterval(() => poll(runId), POLL_INTERVAL);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [runId, poll]);

  const status = jobStatus?.status ?? "queued";
  const progress = jobStatus?.progress ?? 0;
  const message = jobStatus?.message ?? "Waiting…";

  const goToResults = () => {
    router.push(`/dashboard/project/${projectId}/results`);
  };

  // Phase-specific progress using each phase's explicit start/end bounds
  const phaseProgress = (start: number, end: number): number => {
    if (progress <= start) return 0;
    if (progress >= end) return 100;
    return Math.round(((progress - start) / (end - start)) * 100);
  };

  return (
    <div className="max-w-4xl mx-auto pb-20 space-y-8 animate-fade-in">
      {/* Header */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-primary" />
          <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">Step 3 of 3</span>
        </div>
        <h1 className="font-headline text-4xl font-black text-[var(--color-on-surface)] tracking-tighter">Running Tests</h1>
        <p className="text-[var(--color-on-surface-variant)] text-sm font-medium opacity-60">
          {status === "completed" ? "All tests complete!" : status === "failed" ? "Pipeline encountered an error." : "Test pipeline is running. This may take a few minutes."}
        </p>
      </div>

      {/* Overall Progress */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {status === "running" || status === "queued" ? (
              <span className="material-symbols-outlined text-primary animate-spin">progress_activity</span>
            ) : status === "completed" ? (
              <span className="material-symbols-outlined text-secondary">check_circle</span>
            ) : (
              <span className="material-symbols-outlined text-error">error</span>
            )}
            <div>
              <p className="font-headline text-sm font-black">
                {status === "completed" ? "Complete" : status === "failed" ? "Failed" : "Running Pipeline"}
              </p>
              <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">{message}</p>
            </div>
          </div>
          <span className="font-headline text-3xl font-black text-primary">{progress}%</span>
        </div>
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      {/* Failed state */}
      {status === "failed" && jobStatus?.error && (
        <div className="card p-6 border-error/30 bg-error/5">
          <div className="flex items-start gap-3">
            <span className="material-symbols-outlined text-error">error</span>
            <div>
              <p className="font-semibold text-error text-sm mb-1">Pipeline Error</p>
              <pre className="text-xs text-[var(--color-on-surface-variant)] whitespace-pre-wrap break-all">{jobStatus.error.slice(0, 500)}</pre>
            </div>
          </div>
        </div>
      )}

      {/* Phase Cards */}
      <div className="space-y-4">
        {PHASES.map((phase) => {
          const isComplete = completedPhases.has(phase.id);
          const isActive = jobStatus?.current_phase === phase.id;
          const isPending = !isComplete && !isActive;
          const pProgress = phaseProgress(phase.start, phase.end);
          const phaseData = jobStatus?.phase_results?.[phase.id] as PhaseResult | undefined;

          return (
            <div key={phase.id} className={`card p-5 transition-all ${isActive ? "border-primary/30 shadow-lg shadow-primary/5" : ""}`}>
              <div className="flex items-start gap-4">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${phase.bgColor}`}>
                  {isComplete ? (
                    <span className="material-symbols-outlined text-secondary text-lg">check_circle</span>
                  ) : isActive ? (
                    <span className={`material-symbols-outlined text-lg animate-spin ${phase.color}`}>progress_activity</span>
                  ) : (
                    <span className={`material-symbols-outlined text-lg opacity-40 ${phase.color}`}>{phase.icon}</span>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <p className={`font-headline text-sm font-black ${isPending ? "opacity-40" : ""}`}>{phase.label}</p>
                    {isComplete && phaseData && "total" in phaseData && (
                      <div className="flex items-center gap-2">
                        <span className={`text-xs font-black px-2 py-0.5 rounded-full ${
                          (phaseData as PhaseResult).pass_rate >= 70 ? "badge-pass" : "badge-fail"
                        }`}>
                          {(phaseData as PhaseResult).passed}/{(phaseData as PhaseResult).total} passed
                        </span>
                      </div>
                    )}
                    {isComplete && phaseData && !("total" in phaseData) && phase.id === "performance" && (
                      <span className="text-xs font-bold text-secondary">
                        {((phaseData as Record<string, number>).avg_latency ?? 0).toFixed(0)}ms avg
                      </span>
                    )}
                  </div>

                  {(isActive || isComplete) && (
                    <div className="progress-bar-track mb-3">
                      <div className="progress-bar-fill" style={{ width: `${isComplete ? 100 : pProgress}%` }} />
                    </div>
                  )}

                  {/* Live results preview */}
                  {phaseData && "results" in phaseData && (phaseData as PhaseResult).results && (
                    <div className="space-y-1 mt-2">
                      {((phaseData as PhaseResult).results ?? []).slice(0, 8).map((r, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <span className={`material-symbols-outlined text-sm ${r.passed ? "text-secondary" : "text-error"}`}>
                            {r.passed ? "check_circle" : "cancel"}
                          </span>
                          <span className="truncate text-[var(--color-on-surface)] opacity-70">{r.test_name}</span>
                          <span className={`ml-auto font-bold ${r.passed ? "text-secondary" : "text-error"}`}>
                            {(r.score * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                      {((phaseData as PhaseResult).results ?? []).length > 8 && (
                        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-50">
                          +{((phaseData as PhaseResult).results ?? []).length - 8} more tests…
                        </p>
                      )}
                    </div>
                  )}

                  {/* Performance/Load metrics preview */}
                  {isComplete && phaseData && !("results" in phaseData) && phase.id === "performance" && (
                    <div className="grid grid-cols-3 gap-3 mt-2">
                      {[
                        ["Avg", `${((phaseData as Record<string, number>).avg_latency ?? 0).toFixed(0)}ms`],
                        ["P95", `${((phaseData as Record<string, number>).p95_latency ?? 0).toFixed(0)}ms`],
                        ["Errors", `${((phaseData as Record<string, number>).error_rate ?? 0).toFixed(1)}%`],
                      ].map(([k, v]) => (
                        <div key={k} className="p-2 rounded-lg bg-[var(--color-surface-container-low)] text-center">
                          <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">{k}</p>
                          <p className="text-sm font-black">{v}</p>
                        </div>
                      ))}
                    </div>
                  )}
                  {isComplete && phaseData && !("results" in phaseData) && phase.id === "load" && (
                    <div className="grid grid-cols-3 gap-3 mt-2">
                      {[
                        ["Users", (phaseData as Record<string, number>).concurrent_users ?? 0],
                        ["RPS", `${((phaseData as Record<string, number>).requests_per_second ?? 0).toFixed(1)}`],
                        ["Errors", `${((phaseData as Record<string, number>).error_rate ?? 0).toFixed(1)}%`],
                      ].map(([k, v]) => (
                        <div key={k} className="p-2 rounded-lg bg-[var(--color-surface-container-low)] text-center">
                          <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">{k}</p>
                          <p className="text-sm font-black">{v}</p>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* Feedback loop summary */}
                  {isComplete && phaseData && phase.id === "feedback" && (
                    <div className="mt-2 space-y-1.5">
                      {(phaseData as Record<string, unknown>).new_personas_count != null && (
                        <p className="text-xs text-[var(--color-on-surface-variant)]">
                          <span className="font-bold text-[var(--color-tertiary)]">{String((phaseData as Record<string, unknown>).new_personas_count)}</span> new personas generated targeting weak areas
                        </p>
                      )}
                      {(phaseData as Record<string, unknown>).effective_personas != null && (
                        <p className="text-xs text-[var(--color-on-surface-variant)]">
                          <span className="font-bold">{String((phaseData as Record<string, unknown>).effective_personas)}</span> personas found failures and were reinforced
                        </p>
                      )}
                    </div>
                  )}
                  {/* Report ready */}
                  {isComplete && phase.id === "report" && (
                    <p className="mt-2 text-xs text-secondary font-semibold flex items-center gap-1">
                      <span className="material-symbols-outlined text-sm">check_circle</span>
                      HTML report ready — available in Results → Export
                    </p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary + CTA when complete */}
      {status === "completed" && (
        <div className="space-y-6">
          {/* Quick summary */}
          <div className="card p-6">
            <h2 className="font-headline text-base font-black mb-4">Pipeline Complete</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {PHASES.filter(p => !["feedback", "report"].includes(p.id)).map((phase) => {
                const data = jobStatus?.phase_results?.[phase.id];
                if (!data) return null;
                let summary = "";
                if ("total" in data) {
                  const d = data as PhaseResult;
                  summary = `${d.passed}/${d.total} (${d.pass_rate}%)`;
                } else if (phase.id === "performance") {
                  const d = data as Record<string, number>;
                  summary = `${(d.avg_latency ?? 0).toFixed(0)}ms avg`;
                } else if (phase.id === "load") {
                  const d = data as Record<string, number>;
                  summary = `${(d.requests_per_second ?? 0).toFixed(1)} rps`;
                }
                return (
                  <div key={phase.id} className="text-center">
                    <span className={`material-symbols-outlined text-xl ${phase.color}`}>{phase.icon}</span>
                    <p className="text-xs font-semibold mt-1">{phase.label}</p>
                    <p className="text-sm font-black text-[var(--color-on-surface)]">{summary}</p>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="flex justify-center">
            <button onClick={goToResults} className="group flex items-center gap-3 px-10 py-4 rounded-2xl btn-primary text-sm shadow-xl shadow-primary/10">
              <span className="material-symbols-outlined text-xl">bar_chart</span>
              View Detailed Results
              <span className="material-symbols-outlined text-lg group-hover:translate-x-1 transition-transform">arrow_forward</span>
            </button>
          </div>
        </div>
      )}

      {status === "failed" && (
        <div className="flex justify-center gap-4">
          <button
            onClick={() => router.push(`/dashboard/project/${projectId}/configure`)}
            className="px-6 py-3 rounded-xl border border-[var(--color-outline)] text-sm font-semibold hover:bg-[var(--color-surface-variant)] transition-colors"
          >
            Back to Config
          </button>
        </div>
      )}
    </div>
  );
}
