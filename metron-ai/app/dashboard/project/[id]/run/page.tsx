"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";

const API = "";
const POLL_INTERVAL = 2500;

const METRIC_LABELS: Record<string, string> = {
  hallucination: "Hallucination", answer_relevancy: "Answer Relevancy",
  usefulness: "Usefulness", llm_judge: "LLM Judge",
  pii_leakage: "PII Leakage", toxicity: "Toxicity (Output)",
  prompt_injection: "Prompt Injection", bias_fairness: "Bias & Fairness",
  toxic_request: "Toxic Request", attack_resistance: "Attack Resistance",
  geval_overall: "GEval Overall", ragas_faithfulness: "Faithfulness (RAGAS)",
  ragas_answer_relevancy: "Answer Relevancy (RAGAS)",
  ragas_context_recall: "Context Recall (RAGAS)",
  ragas_context_precision: "Context Precision (RAGAS)",
};
function metricLabel(name: string): string {
  if (METRIC_LABELS[name]) return METRIC_LABELS[name];
  if (name.startsWith("geval_")) return "GEval " + name.slice(6).replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  return name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// ─────────────────────────────────── Types ────────────────────────────────────

interface LogEvent {
  type: string;
  ts: string;
  content: Record<string, unknown>;
}

interface JobStatus {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  current_phase: string | null;
  phase_results: Record<string, unknown>;
  log_events: LogEvent[];
  error?: string;
}

// ─────────────────────── Event Feed Renderers ─────────────────────────────────

function PhaseHeader({ content }: { content: Record<string, unknown> }) {
  const icons: Record<string, string> = {
    personas: "person",
    test_gen: "edit_note",
    execution: "play_arrow",
    evaluation: "analytics",
    performance: "speed",
    load: "group",
    feedback: "psychology",
    report: "description",
  };
  const phase = content.phase as string;
  return (
    <div className="flex items-center gap-3 py-2">
      <div className="flex-1 h-px bg-[var(--color-outline-variant)]" />
      <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
        <span className="material-symbols-outlined text-sm text-primary">{icons[phase] || "circle"}</span>
        <span className="text-[11px] font-black uppercase tracking-widest text-primary">{content.label as string}</span>
      </div>
      <div className="flex-1 h-px bg-[var(--color-outline-variant)]" />
    </div>
  );
}

function PersonaCard({ content }: { content: Record<string, unknown> }) {
  const intent = content.intent as string;
  const intentColor =
    intent === "adversarial" ? "bg-error/10 text-error border-error/20" :
    intent === "edge_case"   ? "bg-[#855300]/10 text-[#855300] border-[#855300]/20" :
    "bg-primary/10 text-primary border-primary/20";

  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
        <span className="material-symbols-outlined text-sm text-primary">person</span>
      </div>
      <div className="flex-1 p-3 rounded-2xl rounded-tl-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
        <div className="flex items-center gap-2 flex-wrap mb-1.5">
          <p className="text-sm font-black text-[var(--color-on-surface)]">{content.name as string}</p>
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${intentColor}`}>{intent}</span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--color-surface-variant)] font-semibold">{content.expertise as string}</span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--color-surface-variant)] font-semibold">{content.emotional_state as string}</span>
        </div>
        {!!content.goal && (
          <p className="text-xs text-[var(--color-on-surface-variant)] opacity-80 leading-relaxed">
            Goal: {content.goal as string}
          </p>
        )}
        <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-40 mt-1">{content.user_type as string}</p>
      </div>
    </div>
  );
}

function TestPromptCard({ content }: { content: Record<string, unknown> }) {
  const isSecure = content.test_class === "security";
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${isSecure ? "bg-error/10" : "bg-secondary/10"}`}>
        <span className={`material-symbols-outlined text-sm ${isSecure ? "text-error" : "text-secondary"}`}>
          {isSecure ? "security" : "edit_note"}
        </span>
      </div>
      <div className="flex-1 p-3 rounded-2xl rounded-tl-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
          <span className={`text-[10px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full ${isSecure ? "bg-error/10 text-error" : "bg-secondary/10 text-secondary"}`}>
            {content.test_class as string}
          </span>
          <span className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">for {content.persona_name as string}</span>
          {isSecure && !!content.attack_category && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#855300]/10 text-[#855300] font-semibold">{content.attack_category as string}</span>
          )}
        </div>
        <p className="text-xs font-medium text-[var(--color-on-surface)] leading-relaxed">"{content.text as string}"</p>
        {!isSecure && !!content.expected_behavior && (
          <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60 mt-1.5">
            Expected: {content.expected_behavior as string}
          </p>
        )}
      </div>
    </div>
  );
}

function QualityCriteriaCard({ content }: { content: Record<string, unknown> }) {
  const criteria = content.criteria as string[];
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-secondary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
        <span className="material-symbols-outlined text-sm text-secondary">grade</span>
      </div>
      <div className="flex-1 p-3 rounded-2xl rounded-tl-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
        <p className="text-xs font-black mb-2">Domain quality criteria for <span className="text-primary capitalize">{content.domain as string}</span></p>
        <div className="flex flex-wrap gap-1.5">
          {criteria.map((c, i) => (
            <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-secondary/10 text-secondary font-semibold">{c}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function ConversationCard({ content }: { content: Record<string, unknown> }) {
  const testClass = content.test_class as string;
  const latency = content.latency_ms as number;
  const done = content.done as number;
  const total = content.total as number;

  const classColor =
    testClass === "security" ? "text-error" :
    testClass === "quality"  ? "text-secondary" : "text-primary";

  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)] flex items-center justify-center flex-shrink-0 mt-0.5">
        <span className={`material-symbols-outlined text-sm ${classColor}`}>chat</span>
      </div>
      <div className="flex-1 space-y-2">
        {/* Header */}
        <div className="flex items-center gap-2 text-[10px] text-[var(--color-on-surface-variant)] opacity-60">
          <span className="font-bold text-[var(--color-on-surface)] opacity-100">{content.persona_name as string}</span>
          <span>·</span>
          <span className={`font-semibold uppercase ${classColor}`}>{testClass}</span>
          <span>·</span>
          <span>{latency}ms</span>
          <span className="ml-auto">{done}/{total}</span>
        </div>
        {/* Query bubble */}
        <div className="p-3 rounded-2xl rounded-tl-sm bg-primary/5 border border-primary/10">
          <p className="text-[10px] font-bold text-primary mb-0.5 uppercase tracking-wider">User asked</p>
          <p className="text-xs text-[var(--color-on-surface)] leading-relaxed">{content.query as string}</p>
        </div>
        {/* Response bubble */}
        <div className="p-3 rounded-2xl rounded-tr-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)] ml-4">
          <p className="text-[10px] font-bold text-[var(--color-on-surface-variant)] mb-0.5 uppercase tracking-wider opacity-60">AI responded</p>
          <p className="text-xs text-[var(--color-on-surface)] leading-relaxed opacity-80">{content.response as string}</p>
        </div>
      </div>
    </div>
  );
}

function EvalBatchCard({ content }: { content: Record<string, unknown> }) {
  const phase = content.phase as string;
  const total = content.total as number;
  const passed = content.passed as number;
  const avgScore = content.avg_score as number;
  const samples = (content.samples as Array<{ metric_name: string; persona_name: string; score: number; passed: boolean; reason: string }>) || [];

  const phaseColor =
    phase === "security" ? "text-error bg-error/10 border-error/20" :
    phase === "quality"  ? "text-secondary bg-secondary/10 border-secondary/20" :
    "text-primary bg-primary/10 border-primary/20";

  const passRate = total > 0 ? Math.round((passed / total) * 100) : 0;

  return (
    <div className="flex gap-3 animate-fade-in">
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${phaseColor.split(" ").slice(1).join(" ")}`}>
        <span className={`material-symbols-outlined text-sm ${phaseColor.split(" ")[0]}`}>analytics</span>
      </div>
      <div className="flex-1 p-3 rounded-2xl rounded-tl-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
        <div className="flex items-center gap-3 mb-2">
          <span className={`text-[10px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full border ${phaseColor}`}>{phase} eval</span>
          <span className={`text-sm font-black ${passRate >= 70 ? "text-secondary" : "text-error"}`}>{passRate}%</span>
          <span className="text-xs text-[var(--color-on-surface-variant)] opacity-60">{passed}/{total} passed · avg {avgScore}%</span>
        </div>
        {/* Sample results */}
        <div className="space-y-1.5">
          {samples.map((s, i) => (
            <div key={i} className="flex items-start gap-2">
              <span className={`material-symbols-outlined text-sm flex-shrink-0 mt-0.5 ${s.passed ? "text-secondary" : "text-error"}`}>
                {s.passed ? "check_circle" : "cancel"}
              </span>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[11px] font-semibold text-[var(--color-on-surface)]">{metricLabel(s.metric_name)}</span>
                  <span className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60">— {s.persona_name}</span>
                  <span className={`text-[11px] font-black ml-auto ${s.passed ? "text-secondary" : "text-error"}`}>{s.score}%</span>
                </div>
                {s.reason && (
                  <p className="text-[10px] text-[var(--color-on-surface-variant)] opacity-60 leading-relaxed truncate">{s.reason}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PerfCard({ content }: { content: Record<string, unknown> }) {
  const metrics = [
    { label: "Avg Latency", value: `${content.avg_latency_ms as number}ms` },
    { label: "P95 Latency", value: `${content.p95_latency_ms as number}ms` },
    { label: "P99 Latency", value: `${content.p99_latency_ms as number}ms` },
    { label: "Error Rate",  value: `${content.error_rate as number}%` },
    { label: "Throughput",  value: `${content.throughput_rps as number} rps` },
    { label: "Requests",    value: `${content.successful as number}/${content.total_requests as number}` },
  ];
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-[#855300]/10 flex items-center justify-center flex-shrink-0 mt-0.5">
        <span className="material-symbols-outlined text-sm text-[#855300]">speed</span>
      </div>
      <div className="flex-1 p-3 rounded-2xl rounded-tl-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
        <p className="text-[10px] font-black uppercase tracking-wider text-[#855300] mb-2">Performance Results</p>
        <div className="grid grid-cols-3 gap-2">
          {metrics.map(({ label, value }) => (
            <div key={label} className="text-center p-2 rounded-lg bg-[var(--color-surface-variant)]/50">
              <p className="text-[9px] text-[var(--color-on-surface-variant)] opacity-60 uppercase tracking-wider">{label}</p>
              <p className="text-sm font-black text-[var(--color-on-surface)]">{value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function LoadCard({ content }: { content: Record<string, unknown> }) {
  const errorRate = content.error_rate as number;
  const assessment = content.assessment as string;
  const assessColor = assessment === "excellent" ? "text-secondary" : assessment === "acceptable" ? "text-[#855300]" : "text-error";
  const metrics = [
    { label: "Users",     value: content.concurrent_users as number },
    { label: "Requests",  value: `${content.successful as number}/${content.total_requests as number}` },
    { label: "RPS",       value: `${content.requests_per_second as number}` },
    { label: "Avg",       value: `${content.avg_latency_ms as number}ms` },
    { label: "P95",       value: `${content.p95_latency_ms as number}ms` },
    { label: "Errors",    value: `${errorRate}%` },
  ];
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-[var(--color-surface-variant)] flex items-center justify-center flex-shrink-0 mt-0.5">
        <span className="material-symbols-outlined text-sm text-[var(--color-on-surface-variant)]">group</span>
      </div>
      <div className="flex-1 p-3 rounded-2xl rounded-tl-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
        <div className="flex items-center gap-2 mb-2">
          <p className="text-[10px] font-black uppercase tracking-wider text-[var(--color-on-surface-variant)]">Load Test Results</p>
          {assessment && <span className={`text-[10px] font-bold capitalize ${assessColor}`}>— {assessment}</span>}
        </div>
        <div className="grid grid-cols-3 gap-2">
          {metrics.map(({ label, value }) => (
            <div key={label} className="text-center p-2 rounded-lg bg-[var(--color-surface-variant)]/50">
              <p className="text-[9px] text-[var(--color-on-surface-variant)] opacity-60 uppercase tracking-wider">{label}</p>
              <p className="text-sm font-black text-[var(--color-on-surface)]">{String(value)}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function FeedbackCard({ content }: { content: Record<string, unknown> }) {
  const newNames = (content.new_persona_names as string[]) || [];
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-[var(--color-tertiary-container)]/30 flex items-center justify-center flex-shrink-0 mt-0.5">
        <span className="material-symbols-outlined text-sm text-[var(--color-tertiary)]">psychology</span>
      </div>
      <div className="flex-1 p-3 rounded-2xl rounded-tl-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)]">
        <p className="text-[10px] font-black uppercase tracking-wider text-[var(--color-tertiary)] mb-1.5">Adaptive Feedback Complete</p>
        <p className="text-xs text-[var(--color-on-surface-variant)]">
          <span className="font-bold text-[var(--color-on-surface)]">{content.new_personas_count as number}</span> new targeted personas generated ·{" "}
          <span className="font-bold text-[var(--color-on-surface)]">{content.effective_personas as number}</span> weak areas identified
        </p>
        {newNames.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {newNames.map((n, i) => (
              <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--color-tertiary-container)]/30 font-semibold">{n}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function PipelineCompleteCard({ content }: { content: Record<string, unknown> }) {
  const score = content.health_score as number;
  const passed = content.passed as boolean;
  const scoreColor = score >= 70 ? "text-secondary" : score >= 50 ? "text-[#855300]" : "text-error";
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-secondary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
        <span className="material-symbols-outlined text-sm text-secondary">check_circle</span>
      </div>
      <div className="flex-1 p-4 rounded-2xl rounded-tl-sm bg-secondary/5 border border-secondary/20">
        <div className="flex items-center gap-4">
          <div>
            <p className="text-[10px] font-black uppercase tracking-wider text-secondary mb-1">Pipeline Complete</p>
            <p className="text-xs text-[var(--color-on-surface-variant)]">
              {content.total_passed as number}/{content.total_tests as number} tests passed · Domain: <span className="capitalize font-semibold">{content.domain as string}</span>
            </p>
          </div>
          <div className="ml-auto text-right">
            <p className={`font-headline text-3xl font-black ${scoreColor}`}>{score}%</p>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${passed ? "badge-pass" : "badge-fail"}`}>
              {passed ? "PASSED" : "FAILED"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function EventItem({ event }: { event: LogEvent }) {
  const { type, content } = event;
  if (type === "phase_start")       return <PhaseHeader content={content} />;
  if (type === "persona_created")   return <PersonaCard content={content} />;
  if (type === "test_prompt")       return <TestPromptCard content={content} />;
  if (type === "quality_criteria")  return <QualityCriteriaCard content={content} />;
  if (type === "conversation")      return <ConversationCard content={content} />;
  if (type === "eval_batch")        return <EvalBatchCard content={content} />;
  if (type === "perf_complete")     return <PerfCard content={content} />;
  if (type === "load_complete")     return <LoadCard content={content} />;
  if (type === "feedback_complete") return <FeedbackCard content={content} />;
  if (type === "pipeline_complete") return <PipelineCompleteCard content={content} />;
  return null;
}

// ─────────────────────────────── Main Page ────────────────────────────────────

export default function RunPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [runId, setRunId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const feedBottomRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
      if (data.status === "completed" || data.status === "failed") {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch (_) {}
  }, []);

  useEffect(() => {
    if (!runId) return;
    poll(runId);
    pollRef.current = setInterval(() => poll(runId), POLL_INTERVAL);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [runId, poll]);

  // Auto-scroll feed to bottom when new events arrive
  useEffect(() => {
    feedBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [jobStatus?.log_events?.length]);

  const status = jobStatus?.status ?? "queued";
  const progress = jobStatus?.progress ?? 0;
  const message = jobStatus?.message ?? "Waiting…";
  const events = jobStatus?.log_events ?? [];

  return (
    <div className="max-w-3xl mx-auto pb-20 space-y-6 animate-fade-in">
      {/* Header */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-primary" />
          <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">Step 3 of 3</span>
        </div>
        <h1 className="font-headline text-4xl font-black text-[var(--color-on-surface)] tracking-tighter">Running Tests</h1>
        <p className="text-[var(--color-on-surface-variant)] text-sm font-medium opacity-60">
          {status === "completed" ? "All tests complete — click below to see results." :
           status === "failed"    ? "Pipeline encountered an error." :
           "Watch the pipeline work in real time below."}
        </p>
      </div>

      {/* Sticky progress bar */}
      <div className="card p-5 space-y-3">
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
                {status === "completed" ? "Pipeline Complete" : status === "failed" ? "Failed" : "Running Pipeline"}
              </p>
              <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60 max-w-xs truncate">{message}</p>
            </div>
          </div>
          <span className="font-headline text-3xl font-black text-primary">{progress}%</span>
        </div>
        <div className="progress-bar-track">
          <div className="progress-bar-fill transition-all duration-700" style={{ width: `${progress}%` }} />
        </div>
      </div>

      {/* Error */}
      {status === "failed" && jobStatus?.error && (
        <div className="card p-5 border-error/30 bg-error/5">
          <div className="flex items-start gap-3">
            <span className="material-symbols-outlined text-error">error</span>
            <div>
              <p className="font-semibold text-error text-sm mb-1">Pipeline Error</p>
              <pre className="text-xs text-[var(--color-on-surface-variant)] whitespace-pre-wrap break-all">{jobStatus.error.slice(0, 500)}</pre>
            </div>
          </div>
        </div>
      )}

      {/* Queued / empty state */}
      {events.length === 0 && status !== "failed" && (
        <div className="flex flex-col items-center justify-center py-20 gap-3 text-[var(--color-on-surface-variant)]">
          <span className="material-symbols-outlined text-4xl animate-spin text-primary">progress_activity</span>
          <p className="text-sm opacity-60">Initializing pipeline…</p>
        </div>
      )}

      {/* Live feed */}
      {events.length > 0 && (
        <div className="space-y-3">
          {/* Timestamp header */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-40">Live Activity</span>
            <div className="flex-1 h-px bg-[var(--color-outline-variant)]" />
            <span className="text-[10px] text-[var(--color-on-surface-variant)] opacity-40">{events.length} events</span>
          </div>

          {events.map((ev, i) => (
            <EventItem key={`${ev.type}_${i}`} event={ev} />
          ))}

          {/* Typing indicator when still running */}
          {(status === "running" || status === "queued") && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                <span className="material-symbols-outlined text-sm text-primary animate-spin">progress_activity</span>
              </div>
              <div className="p-3 rounded-2xl rounded-tl-sm bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)] flex items-center gap-2">
                <span className="text-xs text-[var(--color-on-surface-variant)] opacity-60 italic">{message}</span>
              </div>
            </div>
          )}

          <div ref={feedBottomRef} />
        </div>
      )}

      {/* CTA when done */}
      {status === "completed" && (
        <div className="flex justify-center pt-4">
          <button
            onClick={() => router.push(`/dashboard/project/${projectId}/results`)}
            className="group flex items-center gap-3 px-10 py-4 rounded-2xl btn-primary text-sm shadow-xl shadow-primary/10"
          >
            <span className="material-symbols-outlined text-xl">bar_chart</span>
            View Detailed Results
            <span className="material-symbols-outlined text-lg group-hover:translate-x-1 transition-transform">arrow_forward</span>
          </button>
        </div>
      )}

      {status === "failed" && (
        <div className="flex justify-center">
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
