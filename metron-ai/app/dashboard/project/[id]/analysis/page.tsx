"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";

// ── Types matching the backend report structure ───────────────────────────────

interface MetricSummary {
  total: number;
  passed: number;
  pass_rate: number;
  avg_score: number;
}

interface ClassSummary {
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  avg_score: number;
  min_score: number;
  by_metric: Record<string, MetricSummary>;
  failures: FailureItem[];
  // security-specific
  pii_leaks?: number;
  injection_failures?: number;
  jailbreak_failures?: number;
}

interface FailureItem {
  persona: string;
  metric_name: string;
  score: number;
  reason: string;
  conversation_id?: string;
  prompt?: string;
  response?: string;
  turns?: { query: string; response: string; latency_ms: number }[];
}

interface PersonaBreakdown {
  persona_name: string;
  user_type: string;
  intent: string;
  total: number;
  passed: number;
  avg_score: number;
  pass_rate: number;
}

interface Report {
  run_id: string;
  health_score: number;
  passed: boolean;
  domain: string;
  application_type: string;
  total_tests: number;
  total_passed: number;
  total_failed: number;
  test_classes: Record<string, ClassSummary>;
  persona_breakdown: PersonaBreakdown[];
  failure_drill_down: FailureItem[];
}

interface JobStatus {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  error?: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const BACKEND = "";

const pct = (n: number) => `${Math.round(n * 100)}%`;
const score2pct = (n: number | undefined) =>
  n != null ? `${Math.round(n * 100)}` : "—";

const scoreColor = (s: number) => {
  if (s >= 0.7) return "text-[#006e2f]";
  if (s >= 0.4) return "text-amber-500";
  return "text-red-500";
};

const SUPERSET_ICONS: Record<string, string> = {
  functional: "fact_check",
  security: "security",
  performance: "speed",
  load: "monitoring",
};

const METRIC_LABELS: Record<string, string> = {
  answer_relevancy: "Answer Relevancy",
  faithfulness: "Faithfulness",
  usefulness: "Usefulness",
  hallucination: "Hallucination",
  rouge_l: "ROUGE-L",
  bert_score_f1: "BERTScore F1",
  bias_fairness: "Bias & Fairness",
  toxicity: "Toxicity",
  toxic_request_compliance: "Toxic Request",
  pii_leakage: "PII Leakage",
  prompt_injection: "Prompt Injection",
  jailbreak_resistance: "Jailbreak Resistance",
  latency_ms: "Latency (ms)",
  moderation: "Moderation",
  throughput_rps: "Throughput (RPS)",
  failure_rate: "Failure Rate",
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function AnalysisPage() {
  const params = useParams();
  const runId = params?.id as string;

  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [expandedFailure, setExpandedFailure] = useState<number | null>(null);
  const [polling, setPolling] = useState(true);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND}/api/job/${runId}/status`);
      if (!res.ok) return;
      const data: JobStatus = await res.json();
      setJobStatus(data);

      if (data.status === "completed") {
        setPolling(false);
        const rRes = await fetch(`${BACKEND}/api/job/${runId}/results`);
        if (rRes.ok) setReport(await rRes.json());
      } else if (data.status === "failed") {
        setPolling(false);
      }
    } catch {
      // network error — keep polling
    }
  }, [runId]);

  useEffect(() => {
    if (!runId || !polling) return;
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [runId, polling, fetchStatus]);

  const healthPct = report ? Math.round(report.health_score * 100) : 0;
  const healthColor =
    healthPct >= 70 ? "#006e2f" : healthPct >= 40 ? "#b45309" : "#ba1a1a";

  // ── Processing View ─────────────────────────────────────────────────────────
  if (!report) {
    const progress = jobStatus?.progress ?? 0;
    const message = jobStatus?.message ?? "Connecting to pipeline...";
    const isFailed = jobStatus?.status === "failed";

    return (
      <div className="h-[calc(100vh-140px)] flex flex-col items-center justify-center gap-8 animate-fade-in">
        {isFailed ? (
          <div className="text-center space-y-4 max-w-md">
            <div className="w-20 h-20 rounded-full bg-red-500/10 flex items-center justify-center mx-auto">
              <span className="material-symbols-outlined text-4xl text-red-500">error</span>
            </div>
            <h2 className="font-headline font-black text-2xl text-on-surface">Pipeline Failed</h2>
            <p className="text-sm text-outline">{jobStatus?.error || "Unknown error"}</p>
          </div>
        ) : (
          <>
            <div className="w-[280px] h-[280px] rounded-full border-2 border-primary/10 flex items-center justify-center relative">
              <div className="absolute inset-0 border-t-4 border-primary rounded-full animate-spin" />
              <div className="absolute inset-4 border-t-2 border-primary/30 rounded-full animate-spin" style={{ animationDirection: "reverse", animationDuration: "3s" }} />
              <div className="text-center z-10">
                <p className="text-6xl font-black font-headline text-on-surface">{progress}%</p>
                <p className="text-[10px] font-black uppercase tracking-[.3em] text-primary mt-1">Running</p>
              </div>
            </div>
            <div className="text-center space-y-2">
              <p className="text-sm font-bold text-on-surface">{message}</p>
              <p className="text-[10px] font-black uppercase tracking-widest text-outline opacity-50">
                Run ID: {runId}
              </p>
            </div>
            <div className="w-80 h-1.5 rounded-full bg-outline-variant/20 overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-1000"
                style={{ width: `${progress}%` }}
              />
            </div>
          </>
        )}
      </div>
    );
  }

  // ── Results View ────────────────────────────────────────────────────────────
  const tabs = ["overview", ...Object.keys(report.test_classes), "failures", "personas"];

  return (
    <div className="h-[calc(100vh-140px)] flex flex-col -m-8 relative bg-[var(--color-surface-container-low)] overflow-hidden">

      {/* ─── Top Bar ─────────────────────────────────────────────────────── */}
      <div className="h-16 bg-[var(--color-surface-container-lowest)] border-b border-outline-variant/10 flex items-center justify-between px-8 gap-4 shrink-0 z-40">
        <div className="flex items-center gap-6 overflow-x-auto no-scrollbar">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`text-[10px] font-black uppercase tracking-widest px-5 py-2 rounded-xl transition-all whitespace-nowrap ${
                activeTab === tab
                  ? "bg-primary text-white shadow-md"
                  : "text-outline hover:bg-surface-container-low"
              }`}
            >
              {METRIC_LABELS[tab] || tab}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-4 shrink-0">
          <div className="text-right">
            <p className="text-2xl font-black font-headline" style={{ color: healthColor }}>
              {healthPct}
              <span className="text-[11px] font-bold text-outline opacity-50">/100</span>
            </p>
            <p className="text-[8px] font-black uppercase tracking-widest text-outline opacity-40">Health Score</p>
          </div>
          <div className={`px-3 py-1.5 rounded-full text-[9px] font-black uppercase tracking-widest ${
            report.passed ? "bg-green-500/10 text-green-700 border border-green-500/20" : "bg-red-500/10 text-red-600 border border-red-500/20"
          }`}>
            {report.passed ? "PASS" : "FAIL"}
          </div>
        </div>
      </div>

      {/* ─── Content ─────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto no-scrollbar p-8">
        <div className="max-w-7xl mx-auto space-y-8 pb-20">

          {/* ── OVERVIEW TAB ─────────────────────────────────────────────── */}
          {activeTab === "overview" && (
            <div className="space-y-8 animate-fade-in">
              {/* Summary cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
                {[
                  { label: "Total Tests", value: report.total_tests, icon: "science" },
                  { label: "Passed", value: report.total_passed, icon: "check_circle", color: "#006e2f" },
                  { label: "Failed", value: report.total_failed, icon: "cancel", color: "#ba1a1a" },
                  { label: "Pass Rate", value: pct(report.total_passed / Math.max(report.total_tests, 1)), icon: "percent" },
                ].map((c) => (
                  <div key={c.label} className="p-6 rounded-3xl bg-[var(--color-surface-container-lowest)] border border-outline-variant/10 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="material-symbols-outlined text-lg text-outline/50">{c.icon}</span>
                      <p className="text-[9px] font-black uppercase tracking-widest text-outline opacity-60">{c.label}</p>
                    </div>
                    <p className="text-3xl font-black font-headline" style={c.color ? { color: c.color } : {}}>
                      {c.value}
                    </p>
                  </div>
                ))}
              </div>

              {/* Per-class health bars */}
              <div className="p-8 rounded-3xl bg-[var(--color-surface-container-lowest)] border border-outline-variant/10 space-y-6">
                <h3 className="font-headline font-black text-xl text-on-surface">Test Suite Results</h3>
                {Object.entries(report.test_classes).map(([cls, summary]) => {
                  const passRate = summary.pass_rate;
                  return (
                    <div key={cls} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="material-symbols-outlined text-primary text-lg">{SUPERSET_ICONS[cls] || "fact_check"}</span>
                          <span className="text-sm font-black capitalize">{cls}</span>
                        </div>
                        <div className="flex items-center gap-4 text-[10px] font-bold text-outline">
                          <span>{summary.passed}/{summary.total} passed</span>
                          <span className="font-black" style={{ color: passRate >= 0.7 ? "#006e2f" : passRate >= 0.4 ? "#b45309" : "#ba1a1a" }}>
                            {pct(passRate)}
                          </span>
                        </div>
                      </div>
                      <div className="h-2 rounded-full bg-outline-variant/20 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{
                            width: pct(passRate),
                            backgroundColor: passRate >= 0.7 ? "#006e2f" : passRate >= 0.4 ? "#b45309" : "#ba1a1a",
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Run metadata */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                {[
                  { label: "Domain", value: report.domain },
                  { label: "AI Type", value: report.application_type },
                  { label: "Run ID", value: report.run_id.slice(0, 8) + "..." },
                ].map((m) => (
                  <div key={m.label} className="p-4 rounded-2xl bg-[var(--color-surface-container-lowest)] border border-outline-variant/10">
                    <p className="text-[9px] font-black uppercase tracking-widest text-outline opacity-50">{m.label}</p>
                    <p className="text-sm font-black text-on-surface mt-1 capitalize">{m.value}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── PER-CLASS TABS (functional / security / performance / load) ── */}
          {report.test_classes[activeTab] && (
            <div className="space-y-6 animate-fade-in">
              {/* Metric breakdown */}
              <div className="p-8 rounded-3xl bg-[var(--color-surface-container-lowest)] border border-outline-variant/10">
                <h3 className="font-headline font-black text-xl text-on-surface mb-6 capitalize">{activeTab} Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(report.test_classes[activeTab].by_metric || {}).map(([metric, ms]) => (
                    <div key={metric} className="flex items-center justify-between p-4 rounded-2xl bg-[var(--color-surface-container-low)] border border-outline-variant/10">
                      <div>
                        <p className="text-xs font-black text-on-surface">{METRIC_LABELS[metric] || metric}</p>
                        <p className="text-[9px] text-outline opacity-50 mt-0.5">{ms.passed}/{ms.total} passed</p>
                      </div>
                      <div className="text-right">
                        <p className={`text-xl font-black font-headline ${scoreColor(ms.avg_score)}`}>
                          {score2pct(ms.avg_score)}
                          <span className="text-[10px] font-bold text-outline opacity-40">%</span>
                        </p>
                        <p className="text-[9px] font-black text-outline opacity-40 uppercase tracking-wider">{pct(ms.pass_rate)} pass</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Security-specific counts */}
              {activeTab === "security" && report.test_classes.security && (
                <div className="grid grid-cols-3 gap-4">
                  {[
                    { label: "PII Leaks", value: report.test_classes.security.pii_leaks ?? 0, bad: true },
                    { label: "Injection Failures", value: report.test_classes.security.injection_failures ?? 0, bad: true },
                    { label: "Jailbreak Failures", value: report.test_classes.security.jailbreak_failures ?? 0, bad: true },
                  ].map((c) => (
                    <div key={c.label} className={`p-6 rounded-3xl border text-center ${c.value > 0 ? "bg-red-500/5 border-red-500/20" : "bg-green-500/5 border-green-500/20"}`}>
                      <p className={`text-3xl font-black font-headline ${c.value > 0 ? "text-red-500" : "text-[#006e2f]"}`}>{c.value}</p>
                      <p className="text-[9px] font-black uppercase tracking-widest text-outline opacity-60 mt-1">{c.label}</p>
                    </div>
                  ))}
                </div>
              )}

              {/* Class failures */}
              {report.test_classes[activeTab].failures?.length > 0 && (
                <div className="p-8 rounded-3xl bg-[var(--color-surface-container-lowest)] border border-outline-variant/10">
                  <h3 className="font-headline font-black text-lg text-on-surface mb-4">Failures in this suite</h3>
                  <div className="space-y-2">
                    {report.test_classes[activeTab].failures.slice(0, 10).map((f, i) => (
                      <div key={i} className="p-4 rounded-2xl bg-red-500/5 border border-red-500/15">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-black text-on-surface">{f.persona}</span>
                          <div className="flex items-center gap-3">
                            <span className="text-[9px] font-bold text-outline bg-surface-container-low px-2 py-0.5 rounded">{METRIC_LABELS[f.metric_name] || f.metric_name}</span>
                            <span className="text-xs font-black text-red-500">{score2pct(f.score)}%</span>
                          </div>
                        </div>
                        <p className="text-[10px] text-outline opacity-60 mt-1 line-clamp-2">{f.reason}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── FAILURES TAB ─────────────────────────────────────────────── */}
          {activeTab === "failures" && (
            <div className="space-y-4 animate-fade-in">
              <h3 className="font-headline font-black text-xl text-on-surface">Worst Failures (Top 20)</h3>
              {report.failure_drill_down.length === 0 ? (
                <div className="p-10 rounded-3xl bg-green-500/5 border border-green-500/20 text-center">
                  <span className="material-symbols-outlined text-4xl text-green-600">check_circle</span>
                  <p className="font-headline font-black text-xl text-green-700 mt-2">No failures detected!</p>
                </div>
              ) : (
                report.failure_drill_down.map((f, i) => (
                  <div key={i} className="rounded-3xl bg-[var(--color-surface-container-lowest)] border border-outline-variant/10 overflow-hidden">
                    <button
                      className="w-full flex items-center justify-between p-6 text-left hover:bg-outline-variant/5 transition-all"
                      onClick={() => setExpandedFailure(expandedFailure === i ? null : i)}
                    >
                      <div className="flex items-center gap-4">
                        <span className={`w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-black ${
                          (f.score ?? 1) < 0.3 ? "bg-red-500/10 text-red-500" : "bg-amber-500/10 text-amber-600"
                        }`}>
                          {i + 1}
                        </span>
                        <div>
                          <p className="text-sm font-black text-on-surface">{f.persona}</p>
                          <p className="text-[9px] font-bold text-outline opacity-50">
                            {(f as any).superset} · {METRIC_LABELS[f.metric_name] || f.metric_name}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-xl font-black font-headline text-red-500">{score2pct(f.score)}%</span>
                        <span className="material-symbols-outlined text-outline text-lg">
                          {expandedFailure === i ? "expand_less" : "expand_more"}
                        </span>
                      </div>
                    </button>

                    {expandedFailure === i && (
                      <div className="px-6 pb-6 space-y-4 border-t border-outline-variant/10 pt-4">
                        <p className="text-xs text-outline">{f.reason}</p>
                        {f.turns && f.turns.length > 0 && (
                          <div className="space-y-2">
                            <p className="text-[9px] font-black uppercase tracking-widest text-outline opacity-50">Conversation</p>
                            {f.turns.map((t, j) => (
                              <div key={j} className="space-y-1">
                                <div className="flex items-start gap-2">
                                  <span className="text-[8px] font-black text-primary uppercase w-12 shrink-0 mt-0.5">User</span>
                                  <p className="text-[10px] text-on-surface bg-outline-variant/10 rounded-xl px-3 py-2 flex-1">{t.query}</p>
                                </div>
                                <div className="flex items-start gap-2">
                                  <span className="text-[8px] font-black text-outline uppercase w-12 shrink-0 mt-0.5">AI</span>
                                  <p className="text-[10px] text-on-surface bg-primary/5 rounded-xl px-3 py-2 flex-1">{t.response}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {/* ── PERSONAS TAB ─────────────────────────────────────────────── */}
          {activeTab === "personas" && (
            <div className="space-y-4 animate-fade-in">
              <h3 className="font-headline font-black text-xl text-on-surface">Persona Performance</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {report.persona_breakdown.map((p, i) => (
                  <div key={i} className="p-6 rounded-3xl bg-[var(--color-surface-container-lowest)] border border-outline-variant/10 space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-black text-on-surface">{p.persona_name}</p>
                        <p className="text-[9px] text-outline opacity-50 capitalize">{p.user_type} · {p.intent}</p>
                      </div>
                      <p className={`text-2xl font-black font-headline ${scoreColor(p.avg_score)}`}>
                        {score2pct(p.avg_score)}%
                      </p>
                    </div>
                    <div className="h-1.5 rounded-full bg-outline-variant/20 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: pct(p.avg_score),
                          backgroundColor: p.avg_score >= 0.7 ? "#006e2f" : p.avg_score >= 0.4 ? "#b45309" : "#ba1a1a",
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[9px] text-outline opacity-50 font-bold">
                      <span>{p.passed}/{p.total} passed</span>
                      <span>{pct(p.pass_rate)} pass rate</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
