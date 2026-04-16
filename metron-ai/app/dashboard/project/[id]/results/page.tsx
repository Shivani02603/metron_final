"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";

const API = "http://localhost:8000";

// ── Metric display name mapping ───────────────────────────────────────────────
const METRIC_LABELS: Record<string, string> = {
  // Functional
  hallucination:          "Hallucination",
  answer_relevancy:       "Answer Relevancy",
  usefulness:             "Usefulness",
  llm_judge:              "LLM Judge",
  // Security
  pii_leakage:            "PII Leakage",
  toxicity:               "Toxicity (Output)",
  prompt_injection:       "Prompt Injection",
  bias_fairness:          "Bias & Fairness",
  toxic_request:          "Toxic Request",
  attack_resistance:      "Attack Resistance",
  // Quality
  geval_overall:          "GEval Overall",
  ragas_faithfulness:     "Faithfulness (RAGAS)",
  ragas_answer_relevancy: "Answer Relevancy (RAGAS)",
  ragas_context_recall:   "Context Recall (RAGAS)",
  ragas_context_precision:"Context Precision (RAGAS)",
  // RAG evaluation — RAGAS
  rag_faithfulness:       "Faithfulness (RAGAS)",
  rag_context_recall:     "Context Recall (RAGAS)",
  rag_context_precision:  "Context Precision (RAGAS)",
  // RAG evaluation — DeepEval
  rag_answer_relevancy:   "Answer Relevancy (DeepEval)",
  rag_context_relevancy:  "Context Relevancy (DeepEval)",
};

function metricLabel(name: string): string {
  if (METRIC_LABELS[name]) return METRIC_LABELS[name];
  if (name.startsWith("geval_")) {
    return "GEval " + name.slice(6).replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  }
  return name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// Security metrics: detection = pass means no issue found; resistance = pass means attack was blocked
function isDetectionMetric(name: string) { return ["pii_leakage", "toxicity", "bias_fairness"].includes(name); }
function isResistanceMetric(name: string) { return ["prompt_injection", "attack_resistance", "toxic_request"].includes(name); }

// ─────────────────────────────────── Types ────────────────────────────────────
interface TestResult {
  test_id: string;
  test_name: string;
  category: string;
  input_text: string;
  output_text: string;
  score: number;
  passed: boolean;
  reasoning: string;
  latency_ms: number;
  failure_taxonomy_id?: string;
  failure_taxonomy_label?: string;
  failure_reason?: string;
  details: Record<string, unknown>;
}

interface PhaseSummary {
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  avg_score: number;
  results: TestResult[];
}

interface PerformanceMetrics {
  total_requests: number;
  successful: number;
  errors: number;
  error_rate: number;
  min_latency: number;
  avg_latency: number;
  median_latency: number;
  p95_latency: number;
  p99_latency: number;
  max_latency: number;
  throughput: number;
}

interface LoadMetrics {
  concurrent_users: number;
  duration_seconds: number;
  total_requests: number;
  successful: number;
  errors: number;
  error_rate: number;
  avg_latency: number;
  p95_latency: number;
  requests_per_second: number;
  tool_used: string;
}

interface RCAFinding {
  rank: number;
  id: string;
  label: string;
  reason?: string;
  category: string;
  category_id: string;
  probability: number;
  affected_count: number;
  evidence: string[];
  remediation: string;
}

interface RCAReport {
  total_failed: number;
  total_analyzed: number;
  relevant_points: number;
  filtered_points: number;
  architecture_summary: Record<string, unknown>;
  signal_summary: Record<string, number>;
  top_causes: RCAFinding[];
}

interface FullResults {
  run_id: string;
  health_score: number;
  passed: boolean;
  domain: string;
  agent_name: string;
  config_summary: Record<string, unknown>;
  functional: PhaseSummary;
  security: PhaseSummary;
  quality: PhaseSummary;
  rag?: PhaseSummary;
  performance: PerformanceMetrics;
  load: LoadMetrics;
  rca?: RCAReport;
  personas: Array<{ id: string; name: string; description: string; traits: string[] }>;
  persona_breakdown: Array<{
    persona_id: string;
    persona_name: string;
    total: number;
    passed: number;
    avg_score: number;
    pass_rate: number;
    intent?: string;
    fishbone?: Record<string, string>;
  }>;
  failure_drill_down: TestResult[];
  total_tests: number;
  total_passed: number;
  report_html?: string;
}

// ─────────────────────────────── Component ────────────────────────────────────
export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [results, setResults] = useState<FullResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    const runId = sessionStorage.getItem(`run_id_${projectId}`);
    if (!runId) {
      setError("No run ID found. Please run the test suite first.");
      setLoading(false);
      return;
    }

    fetch(`${API}/api/job/${runId}/results`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        // Attach run_id into results
        setResults({ ...data, run_id: runId });
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [projectId]);

  const downloadJSON = () => {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `metron_results_${Date.now()}.json`;
    a.click();
  };

  const downloadAllPromptsCSV = () => {
    if (!results) return;

    const escape = (s: string) => `"${String(s ?? "").replace(/"/g, '""')}"`;

    const headers = [
      "Suite", "Metric / Test", "Category",
      "Prompt (Input)", "AI Response (Output)",
      "Score", "Passed", "Reasoning", "Latency (ms)",
    ];

    const toRows = (suite: string, items: TestResult[] = []) =>
      items.map((r) => [
        escape(suite),
        escape(r.test_name || ""),
        escape(r.category || ""),
        escape((r.input_text || "").slice(0, 300)),
        escape((r.output_text || "").slice(0, 300)),
        r.score?.toFixed(4) ?? "",
        r.passed ? "PASS" : "FAIL",
        escape(r.reasoning || ""),
        r.latency_ms?.toFixed(0) ?? "",
      ]);

    const rows = [
      ...toRows("Functional", results.functional?.results),
      ...toRows("Security",   results.security?.results),
      ...toRows("Quality",    results.quality?.results),
    ];

    const csv = [headers, ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `metron_all_prompts_${Date.now()}.csv`;
    a.click();
  };

  const downloadCSV = () => {
    if (!results?.functional?.results) return;
    const headers = ["Test ID", "Name", "Category", "Score", "Passed", "Latency(ms)", "Input", "Output"];
    const rows = results.functional.results.map((r) => [
      r.test_id, r.test_name, r.category,
      r.score.toFixed(3), r.passed ? "true" : "false",
      r.latency_ms.toFixed(0),
      `"${r.input_text.slice(0, 100).replace(/"/g, '""')}"`,
      `"${r.output_text.slice(0, 200).replace(/"/g, '""')}"`,
    ]);
    const csv = [headers, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `metron_functional_${Date.now()}.csv`;
    a.click();
  };

  const downloadHTML = () => {
    if (!results?.report_html) return;
    const blob = new Blob([results.report_html], { type: "text/html" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `metron_report_${Date.now()}.html`;
    a.click();
  };

  const downloadMarkdown = () => {
    if (!results) return;
    const r = results;

    const rcaSection = r.rca ? `
## Root Cause Analysis
> Analysed ${r.rca.total_analyzed} results · ${r.rca.total_failed} failures · ${r.rca.relevant_points} relevant failure points (${r.rca.filtered_points} excluded by architecture filter)

### Observed Signals
${Object.entries(r.rca.signal_summary).map(([k, v]) => `- ${k.replace(/_/g, " ")}: ${v}`).join("\n")}

### Top Probable Root Causes
${r.rca.top_causes.map(c => {
  const pct = Math.round(c.probability * 100);
  const level = pct >= 70 ? "🔴 HIGH" : pct >= 45 ? "🟠 MEDIUM" : "🔵 LOW";
  return `#### #${c.rank} [${c.id}] ${c.label}
- **Category:** ${c.category}
- **Probability:** ${pct}% ${level}
- **Affected:** ${c.affected_count} prompt(s)
- **Evidence:** ${c.evidence.slice(0, 2).join("; ")}
- **Fix:** ${c.remediation}`;
}).join("\n\n")}
` : "";

    const md = `# METRON QA Report
Generated: ${new Date().toLocaleString()}
Agent: ${r.agent_name || "—"} | Domain: ${r.domain}
Health Score: ${(r.health_score * 100).toFixed(1)}% | ${r.passed ? "PASSED" : "FAILED"}

## Summary
| Phase | Passed | Total | Pass Rate |
|-------|--------|-------|-----------|
| Functional | ${r.functional?.passed ?? 0} | ${r.functional?.total ?? 0} | ${r.functional?.pass_rate ?? 0}% |
| Security | ${r.security?.passed ?? 0} | ${r.security?.total ?? 0} | ${r.security?.pass_rate ?? 0}% |
| Quality | ${r.quality?.passed ?? 0} | ${r.quality?.total ?? 0} | ${r.quality?.pass_rate ?? 0}% |

## Performance
- Avg Latency: ${(r.performance?.avg_latency ?? 0).toFixed(0)}ms
- P95: ${(r.performance?.p95_latency ?? 0).toFixed(0)}ms
- Throughput: ${(r.performance?.throughput ?? 0).toFixed(2)} req/s
- Error Rate: ${(r.performance?.error_rate ?? 0).toFixed(1)}%

## Load Test
- Concurrent Users: ${r.load?.concurrent_users ?? 0}
- Throughput: ${(r.load?.requests_per_second ?? 0).toFixed(2)} req/s
- Error Rate: ${(r.load?.error_rate ?? 0).toFixed(1)}%
${rcaSection}`;
    const blob = new Blob([md], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `metron_report_${Date.now()}.md`;
    a.click();
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <span className="material-symbols-outlined text-4xl text-primary animate-spin">progress_activity</span>
        <p className="text-sm text-[var(--color-on-surface-variant)] opacity-60">Loading results…</p>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="max-w-4xl mx-auto py-20 text-center space-y-4">
        <span className="material-symbols-outlined text-5xl text-error">error</span>
        <p className="text-lg font-semibold">{error || "Results not found"}</p>
        <button onClick={() => router.push(`/dashboard/project/${projectId}/configure`)} className="px-6 py-3 rounded-xl btn-primary text-sm">
          Start New Test
        </button>
      </div>
    );
  }

  const healthPct = Math.round(results.health_score * 100);
  const healthColor = healthPct >= 70 ? "text-secondary" : healthPct >= 40 ? "text-[#855300]" : "text-error";

  const TABS = [
    "Functional", "Security", "Quality",
    ...(results.rag ? ["RAG"] : []),
    "Performance", "Load Test",
    ...(results.rca ? ["RCA"] : []),
    "Export",
  ];

  return (
    <div className="max-w-5xl mx-auto pb-20 space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row gap-6 items-start justify-between">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-primary" />
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">Results</span>
          </div>
          <h1 className="font-headline text-4xl font-black text-[var(--color-on-surface)] tracking-tighter">
            {results.agent_name || "Test Results"}
          </h1>
          <p className="text-[var(--color-on-surface-variant)] text-sm opacity-60">{results.domain}</p>
        </div>

        {/* Health Score */}
        <div className="card p-6 flex flex-col items-center gap-2 min-w-[140px]">
          <p className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">Health Score</p>
          <p className={`font-headline text-5xl font-black ${healthColor}`}>{healthPct}%</p>
          <span className={`text-xs font-bold px-3 py-1 rounded-full ${results.passed ? "badge-pass" : "badge-fail"}`}>
            {results.passed ? "PASSED" : "FAILED"}
          </span>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {[
          { label: "Functional", icon: "science", color: "text-primary", value: results.functional ? `${results.functional.passed}/${results.functional.total}` : "—", sub: results.functional ? `${results.functional.pass_rate}%` : "" },
          { label: "Security", icon: "security", color: "text-error", value: results.security ? `${results.security.passed}/${results.security.total}` : "—", sub: results.security ? `${results.security.pass_rate}%` : "" },
          { label: "Quality", icon: "grade", color: "text-secondary", value: results.quality ? `${results.quality.passed}/${results.quality.total}` : "—", sub: results.quality ? `${results.quality.pass_rate}%` : "" },
          { label: "Performance", icon: "speed", color: "text-[#855300]", value: results.performance ? `${(results.performance.avg_latency ?? 0).toFixed(0)}ms` : "—", sub: "avg latency" },
          { label: "Load", icon: "group", color: "text-[var(--color-on-surface-variant)]", value: results.load ? `${(results.load.error_rate ?? 0).toFixed(1)}%` : "—", sub: "error rate" },
        ].map((item) => (
          <div key={item.label} className="card p-4 text-center">
            <span className={`material-symbols-outlined text-xl ${item.color}`}>{item.icon}</span>
            <p className="font-headline text-2xl font-black text-[var(--color-on-surface)] mt-1">{item.value}</p>
            <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">{item.sub || item.label}</p>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="card overflow-hidden">
        {/* Tab bar */}
        <div className="flex overflow-x-auto border-b border-[var(--color-outline-variant)]">
          {TABS.map((tab, i) => (
            <button
              key={tab}
              onClick={() => setActiveTab(i)}
              className={`tab-btn flex-shrink-0 ${activeTab === i ? "active" : ""}`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="p-6">
          {/* ── Tab 0: Functional ── */}
          {activeTab === 0 && <FunctionalTab data={results.functional} personaBreakdown={results.persona_breakdown} />}

          {/* ── Tab 1: Security ── */}
          {activeTab === 1 && <SecurityTab data={results.security} />}

          {/* ── Tab 2: Quality ── */}
          {activeTab === 2 && <QualityTab data={results.quality} />}

          {/* ── Tab 3: RAG (only present in RAG mode) ── */}
          {results.rag && activeTab === 3 && <RAGTab data={results.rag} />}

          {/* ── Performance / Load / RCA / Export — indices shift with optional RAG tab ── */}
          {(() => {
            const base = results.rag ? 4 : 3;
            const rcaIdx   = results.rca ? base + 2 : -1;
            const exportIdx = results.rca ? base + 3 : base + 2;
            return (
              <>
                {activeTab === base     && <PerformanceTab data={results.performance} />}
                {activeTab === base + 1 && <LoadTab data={results.load} />}
                {results.rca && activeTab === rcaIdx && <RCATab data={results.rca} />}
                {activeTab === exportIdx && (
            <div className="space-y-4">
              <p className="text-sm text-[var(--color-on-surface-variant)] opacity-70">Download the test results in various formats.</p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <ExportCard
                  icon="data_object"
                  title="Full JSON Report"
                  description="Complete results with all data, suitable for programmatic processing."
                  onClick={downloadJSON}
                />
                <ExportCard
                  icon="checklist"
                  title="All Prompts CSV"
                  description="Every prompt across functional, security, and quality — with pass/fail, score, AI response, and reasoning."
                  onClick={downloadAllPromptsCSV}
                />
                <ExportCard
                  icon="table_chart"
                  title="Functional CSV"
                  description="Functional test results only as a spreadsheet-compatible CSV file."
                  onClick={downloadCSV}
                />
                {results.report_html && (
                  <ExportCard
                    icon="html"
                    title="HTML Report"
                    description="Styled full report with health score, persona breakdown, and failure drill-down."
                    onClick={downloadHTML}
                  />
                )}
                <ExportCard
                  icon="description"
                  title="Markdown Report"
                  description="Human-readable summary report in Markdown format."
                  onClick={downloadMarkdown}
                />
              </div>
            </div>
                )}
              </>
            );
          })()}
        </div>
      </div>

      {/* CTA */}
      <div className="flex justify-center">
        <button
          onClick={() => router.push(`/dashboard/project/${projectId}/configure`)}
          className="flex items-center gap-2 px-6 py-3 rounded-xl border border-[var(--color-outline)] text-sm font-semibold hover:bg-[var(--color-surface-variant)] transition-colors"
        >
          <span className="material-symbols-outlined text-base">refresh</span>
          Run New Test
        </button>
      </div>
    </div>
  );
}

// ─────────────────────── Functional Tab ────────────────────────────────────
function FunctionalTab({
  data,
  personaBreakdown,
}: {
  data: PhaseSummary;
  personaBreakdown: FullResults["persona_breakdown"];
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  if (!data?.results) return <EmptyState />;

  // Group by category
  const byCategory: Record<string, TestResult[]> = {};
  for (const r of data.results) {
    const cat = r.category || "other";
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push(r);
  }

  const toggle = (id: string) => setExpanded((prev) => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });

  return (
    <div className="space-y-6">
      <SummaryRow4
        items={[
          { label: "Total Tests", value: data.total },
          { label: "Passed", value: data.passed, color: "text-secondary" },
          { label: "Failed", value: data.failed, color: "text-error" },
          { label: "Avg Score", value: `${(data.avg_score * 100).toFixed(1)}%` },
        ]}
      />

      {Object.entries(byCategory).map(([cat, results]) => {
        const catPassed = results.filter((r) => r.passed).length;
        const catRate = Math.round((catPassed / results.length) * 100);
        return (
          <div key={cat} className="border border-[var(--color-outline-variant)] rounded-xl overflow-hidden">
            <button
              onClick={() => toggle(cat)}
              className="w-full flex items-center justify-between p-4 hover:bg-[var(--color-surface-container-low)] transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-base text-primary">category</span>
                <p className="text-sm font-black">{metricLabel(cat)}</p>
              </div>
              <div className="flex items-center gap-3">
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${catRate >= 70 ? "badge-pass" : "badge-fail"}`}>
                  {catPassed}/{results.length} ({catRate}%)
                </span>
                <span className="material-symbols-outlined text-base text-[var(--color-on-surface-variant)]">
                  {expanded.has(cat) ? "expand_less" : "expand_more"}
                </span>
              </div>
            </button>
            {expanded.has(cat) && (
              <div className="divide-y divide-[var(--color-outline-variant)]">
                {results.map((r) => (
                  <TestResultRow key={r.test_id} result={r} />
                ))}
              </div>
            )}
          </div>
        );
      })}

      {personaBreakdown?.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">Persona Breakdown</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {personaBreakdown.map((p) => (
              <div key={p.persona_id} className="p-3 rounded-xl border border-[var(--color-outline-variant)] flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-black truncate">{p.persona_name}</p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {(p.intent || p.fishbone?.intent) && (
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${
                        (p.intent || p.fishbone?.intent) === "adversarial" ? "bg-error/10 text-error" :
                        (p.intent || p.fishbone?.intent) === "edge_case" ? "bg-[#855300]/10 text-[#855300]" :
                        "bg-primary/10 text-primary"
                      }`}>{p.intent || p.fishbone?.intent}</span>
                    )}
                    {p.fishbone?.expertise && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)] font-semibold">
                        {p.fishbone.expertise}
                      </span>
                    )}
                    {p.fishbone?.emotional_state && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)] font-semibold">
                        {p.fishbone.emotional_state}
                      </span>
                    )}
                    {!p.intent && !p.fishbone && (
                      <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">{p.total} tests</p>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-black ${p.pass_rate >= 70 ? "text-secondary" : "text-error"}`}>{p.pass_rate}%</p>
                  <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">{p.passed}/{p.total} passed</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────── Security Tab ─────────────────────────────────────
function SecurityTab({ data }: { data: PhaseSummary }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  if (!data?.results) return <EmptyState />;

  const secScore = data.pass_rate;
  const scoreColor = secScore >= 90 ? "text-secondary" : secScore >= 70 ? "text-[#855300]" : "text-error";
  const toggle = (id: string) => setExpanded((prev) => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });

  // Group by full metric name (category field now holds the full metric_name)
  const byCategory: Record<string, TestResult[]> = {};
  for (const r of data.results) {
    const cat = r.category || "general";
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push(r);
  }

  // Ordered display: detection metrics first, then resistance metrics
  const ORDER = ["pii_leakage", "toxicity", "prompt_injection", "bias_fairness", "toxic_request", "attack_resistance"];
  const sortedEntries = Object.entries(byCategory).sort(([a], [b]) => {
    const ai = ORDER.indexOf(a); const bi = ORDER.indexOf(b);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4 p-4 rounded-xl bg-[var(--color-surface-container-low)]">
        <span className={`font-headline text-4xl font-black ${scoreColor}`}>{secScore.toFixed(1)}%</span>
        <div>
          <p className="font-semibold">Security Score</p>
          <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60">
            {data.passed}/{data.total} checks passed
            {secScore >= 90 ? " — Excellent" : secScore >= 70 ? " — Acceptable" : " — Needs Improvement"}
          </p>
        </div>
      </div>

      {sortedEntries.map(([cat, results]) => {
        const passed = results.filter((r) => r.passed).length;
        const passLabel = isDetectionMetric(cat)
          ? `${passed}/${results.length} clean`
          : `${passed}/${results.length} blocked`;
        const inputLabel = isResistanceMetric(cat) ? "Attack Prompt" : "Input";
        const outputLabel = "AI Response";

        return (
          <div key={cat} className="border border-[var(--color-outline-variant)] rounded-xl overflow-hidden">
            <button
              onClick={() => toggle(cat)}
              className="w-full flex items-center justify-between p-4 hover:bg-[var(--color-surface-container-low)] transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className={`material-symbols-outlined text-base ${passed === results.length ? "text-secondary" : "text-error"}`}>
                  {passed === results.length ? "shield" : "shield_with_warning"}
                </span>
                <p className="text-sm font-black">{metricLabel(cat)}</p>
              </div>
              <div className="flex items-center gap-3">
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${passed === results.length ? "badge-pass" : "badge-fail"}`}>
                  {passLabel}
                </span>
                <span className="material-symbols-outlined text-base text-[var(--color-on-surface-variant)]">
                  {expanded.has(cat) ? "expand_less" : "expand_more"}
                </span>
              </div>
            </button>
            {expanded.has(cat) && (
              <div className="divide-y divide-[var(--color-outline-variant)]">
                {results.map((r) => (
                  <div key={r.test_id} className="p-4 space-y-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`material-symbols-outlined text-base ${r.passed ? "text-secondary" : "text-error"}`}>
                        {r.passed ? "check_circle" : "cancel"}
                      </span>
                      <p className="text-sm font-semibold">{r.test_name}</p>
                      <span className={`text-[10px] font-black ml-auto ${r.passed ? "text-secondary" : "text-error"}`}>
                        {(r.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    {r.reasoning && (
                      <p className="text-xs text-[var(--color-on-surface-variant)] pl-8 leading-relaxed">{r.reasoning}</p>
                    )}
                    {!r.passed && r.failure_reason && (
                      <div className="p-3 rounded-lg border border-[#6200ee]/20 bg-[#6200ee]/5 space-y-1.5">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="material-symbols-outlined text-xs text-[#6200ee]">travel_explore</span>
                          <p className="text-[10px] font-black uppercase tracking-wider text-[#6200ee]">Root Cause</p>
                          {r.failure_taxonomy_id && r.failure_taxonomy_label && (
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#ede7f6] text-[#6200ee]">
                              {r.failure_taxonomy_id} · {r.failure_taxonomy_label}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-[var(--color-on-surface-variant)] leading-relaxed">{r.failure_reason}</p>
                      </div>
                    )}
                    <ConversationBlock
                      input={r.input_text}
                      output={r.output_text}
                      inputLabel={inputLabel}
                      outputLabel={outputLabel}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────── Quality Tab ──────────────────────────────────────
function QualityTab({ data }: { data: PhaseSummary }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  if (!data?.results) return <EmptyState />;

  const toggle = (id: string) => setExpanded((prev) => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });

  return (
    <div className="space-y-6">
      <SummaryRow4
        items={[
          { label: "Total", value: data.total },
          { label: "Passed", value: data.passed, color: "text-secondary" },
          { label: "Failed", value: data.failed, color: "text-error" },
          { label: "Avg Score", value: `${(data.avg_score * 100).toFixed(1)}%` },
        ]}
      />

      {data.results.map((r) => (
        <div key={r.test_id} className="border border-[var(--color-outline-variant)] rounded-xl overflow-hidden">
          <button
            onClick={() => toggle(r.test_id)}
            className="w-full flex items-center justify-between p-4 hover:bg-[var(--color-surface-container-low)] transition-colors"
          >
            <div className="flex items-center gap-3">
              <span className={`material-symbols-outlined text-base ${r.passed ? "text-secondary" : "text-error"}`}>
                {r.passed ? "check_circle" : "cancel"}
              </span>
              <p className="text-sm font-black">{r.test_name}</p>
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-xs font-black ${r.passed ? "text-secondary" : "text-error"}`}>
                {(r.score * 100).toFixed(0)}%
              </span>
              <span className="material-symbols-outlined text-base text-[var(--color-on-surface-variant)]">
                {expanded.has(r.test_id) ? "expand_less" : "expand_more"}
              </span>
            </div>
          </button>
          {expanded.has(r.test_id) && (
            <div className="px-4 pb-4 space-y-3 border-t border-[var(--color-outline-variant)]">
              <ConversationBlock input={r.input_text} output={r.output_text} inputLabel="Question" outputLabel="Response" />
              {r.reasoning && (
                <div className="p-3 rounded-lg bg-[var(--color-surface-container-low)]">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-on-surface-variant)] opacity-60 mb-1">Evaluation</p>
                  <p className="text-xs">{r.reasoning}</p>
                </div>
              )}
              {!r.passed && r.failure_reason && (
                <div className="p-3 rounded-lg border border-[#6200ee]/20 bg-[#6200ee]/5 space-y-1.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="material-symbols-outlined text-xs text-[#6200ee]">travel_explore</span>
                    <p className="text-[10px] font-black uppercase tracking-wider text-[#6200ee]">Root Cause</p>
                    {r.failure_taxonomy_id && r.failure_taxonomy_label && (
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#ede7f6] text-[#6200ee]">
                        {r.failure_taxonomy_id} · {r.failure_taxonomy_label}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-[var(--color-on-surface-variant)] leading-relaxed">{r.failure_reason}</p>
                </div>
              )}
              {/* Metric breakdown from details */}
              {(() => {
                type MetricEntry = { metric_name: string; score: number; passed: boolean };
                const metrics = (r.details as { metrics?: MetricEntry[] })?.metrics;
                if (!Array.isArray(metrics) || metrics.length === 0) return null;
                return (
                  <div className="space-y-2">
                    {metrics.map((m, i) => (
                      <div key={i} className="flex items-center gap-3 text-xs">
                        <span className={`material-symbols-outlined text-sm ${m.passed ? "text-secondary" : "text-error"}`}>
                          {m.passed ? "check" : "close"}
                        </span>
                        <span className="font-semibold capitalize">{m.metric_name.replace(/_/g, " ")}</span>
                        <span className="ml-auto font-black">{(m.score * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ─────────────────────── RAG Tab ──────────────────────────────────────────
function RAGTab({ data }: { data: PhaseSummary }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  if (!data?.results) return <EmptyState />;

  const toggle = (id: string) => setExpanded((prev) => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });

  const RAG_METRIC_ICONS: Record<string, string> = {
    rag_faithfulness:      "verified",
    rag_context_relevancy: "manage_search",
    rag_context_recall:    "library_books",
    rag_context_precision: "target",
  };

  const byMetric = data.results.reduce<Record<string, TestResult[]>>((acc, r) => {
    (acc[r.category] = acc[r.category] || []).push(r);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <SummaryRow4
        items={[
          { label: "Total", value: data.total },
          { label: "Passed", value: data.passed, color: "text-secondary" },
          { label: "Failed", value: data.failed, color: "text-error" },
          { label: "Avg Score", value: `${(data.avg_score * 100).toFixed(1)}%` },
        ]}
      />

      {/* Per-metric summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(byMetric).map(([metricKey, items]) => {
          const avg = items.reduce((s, r) => s + r.score, 0) / items.length;
          const passed = items.filter((r) => r.passed).length;
          return (
            <div key={metricKey} className="card p-4 text-center space-y-1">
              <span className="material-symbols-outlined text-xl text-primary">
                {RAG_METRIC_ICONS[metricKey] || "analytics"}
              </span>
              <p className="font-headline text-2xl font-black text-[var(--color-on-surface)]">{(avg * 100).toFixed(0)}%</p>
              <p className="text-[9px] font-black uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">
                {metricLabel(metricKey)}
              </p>
              <p className="text-[10px] text-secondary font-bold">{passed}/{items.length} passed</p>
            </div>
          );
        })}
      </div>

      {/* Per-question breakdown */}
      <div className="space-y-3">
        {data.results.map((r) => (
          <div key={r.test_id} className="border border-[var(--color-outline-variant)] rounded-xl overflow-hidden">
            <button
              onClick={() => toggle(r.test_id)}
              className="w-full flex items-center justify-between p-4 hover:bg-[var(--color-surface-container-low)] transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                <span className={`material-symbols-outlined text-base flex-shrink-0 ${r.passed ? "text-secondary" : "text-error"}`}>
                  {r.passed ? "check_circle" : "cancel"}
                </span>
                <div className="min-w-0">
                  <p className="text-xs font-black text-[var(--color-on-surface-variant)] opacity-60 uppercase tracking-wider">{metricLabel(r.category)}</p>
                  <p className="text-sm font-bold truncate">{r.input_text.slice(0, 80)}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <span className={`text-xs font-black ${r.passed ? "text-secondary" : "text-error"}`}>{(r.score * 100).toFixed(0)}%</span>
                <span className="material-symbols-outlined text-base text-[var(--color-on-surface-variant)]">
                  {expanded.has(r.test_id) ? "expand_less" : "expand_more"}
                </span>
              </div>
            </button>
            {expanded.has(r.test_id) && (
              <div className="px-4 pb-4 space-y-3 border-t border-[var(--color-outline-variant)]">
                <ConversationBlock input={r.input_text} output={r.output_text} inputLabel="Question" outputLabel="RAG Answer" />
                {r.reasoning && (
                  <div className="p-3 rounded-lg bg-[var(--color-surface-container-low)]">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-on-surface-variant)] opacity-60 mb-1">Evaluation</p>
                    <p className="text-xs">{r.reasoning}</p>
                  </div>
                )}
                {!r.passed && r.failure_reason && (
                  <div className="p-3 rounded-lg border border-[#6200ee]/20 bg-[#6200ee]/5 space-y-1.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="material-symbols-outlined text-xs text-[#6200ee]">travel_explore</span>
                      <p className="text-[10px] font-black uppercase tracking-wider text-[#6200ee]">Root Cause</p>
                      {r.failure_taxonomy_id && r.failure_taxonomy_label && (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#ede7f6] text-[#6200ee]">
                          {r.failure_taxonomy_id} · {r.failure_taxonomy_label}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-[var(--color-on-surface-variant)] leading-relaxed">{r.failure_reason}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────── RCA Tab ──────────────────────────────────────────
function RCATab({ data }: { data: RCAReport }) {
  if (!data?.top_causes?.length) return <EmptyState />;

  const CAT_COLORS: Record<string, { bg: string; text: string }> = {
    C1: { bg: "bg-purple-100",  text: "text-purple-700"  },
    C2: { bg: "bg-blue-100",    text: "text-blue-700"    },
    C3: { bg: "bg-orange-100",  text: "text-orange-700"  },
    C4: { bg: "bg-red-100",     text: "text-red-700"     },
    C5: { bg: "bg-green-100",   text: "text-green-700"   },
    C6: { bg: "bg-yellow-100",  text: "text-yellow-700"  },
    C7: { bg: "bg-fuchsia-100", text: "text-fuchsia-700" },
    C8: { bg: "bg-slate-100",   text: "text-slate-700"   },
  };

  const arch = data.architecture_summary as Record<string, unknown>;

  return (
    <div className="space-y-6">
      {/* Summary stats */}
      <SummaryRow4
        items={[
          { label: "Failures Mapped",    value: data.total_failed },
          { label: "Results Analysed",   value: data.total_analyzed },
          { label: "Relevant FP",        value: data.relevant_points },
          { label: "Filtered Out",       value: data.filtered_points },
        ]}
      />

      {/* Architecture profile */}
      <div className="p-4 rounded-xl bg-[var(--color-surface-container-low)] space-y-2">
        <p className="text-[10px] font-black uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">Architecture Profile Used</p>
        <div className="flex flex-wrap gap-2">
          {[
            ["App Type",        arch.application_type as string],
            ["Deployment",      arch.deployment_type as string],
            ["Vector DB",       (arch.vector_db as string) || "none"],
            ["Session DB",      (arch.session_db as string) || "none"],
            ["Rate Limiting",   arch.has_rate_limiting ? "yes" : "no"],
            ["Retry Logic",     arch.has_retry_logic ? "yes" : "no"],
            ["Circuit Breaker", arch.has_circuit_breaker ? "yes" : "no"],
          ].map(([label, val]) => {
            const ok = !["no","none","unknown"].includes(String(val).toLowerCase());
            return (
              <span key={label} className={`text-[11px] font-semibold px-2.5 py-1 rounded-full ${ok ? "bg-secondary/10 text-secondary" : "bg-error/10 text-error"}`}>
                {label}: {val}
              </span>
            );
          })}
        </div>
      </div>

      {/* Observed signals */}
      {Object.keys(data.signal_summary).length > 0 && (
        <div className="p-4 rounded-xl bg-[var(--color-surface-container-low)] space-y-2">
          <p className="text-[10px] font-black uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">Observed Failure Signals</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(data.signal_summary)
              .sort(([, a], [, b]) => b - a)
              .map(([key, count]) => (
                <span key={key} className="text-[11px] font-semibold px-2.5 py-1 rounded-full bg-[#855300]/10 text-[#855300]">
                  {key.replace(/_/g, " ")} ({count})
                </span>
              ))}
          </div>
        </div>
      )}

      {/* Root cause cards */}
      <div className="space-y-3">
        <p className="text-[10px] font-black uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60">Top Probable Root Causes</p>
        {data.top_causes.map((cause) => {
          const pct = Math.round(cause.probability * 100);
          const isHigh   = pct >= 70;
          const isMedium = pct >= 45 && pct < 70;
          const probColor = isHigh ? "text-error" : isMedium ? "text-[#855300]" : "text-primary";
          const probBg    = isHigh ? "bg-error/10" : isMedium ? "bg-[#855300]/10" : "bg-primary/10";
          const probLabel = isHigh ? "HIGH" : isMedium ? "MEDIUM" : "LOW";
          const cat = CAT_COLORS[cause.category_id] ?? { bg: "bg-slate-100", text: "text-slate-700" };

          return (
            <div key={cause.id} className="border border-[var(--color-outline-variant)] rounded-xl p-4">
              <div className="flex items-start gap-3">
                {/* Left: details */}
                <div className="flex-1 min-w-0 space-y-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-black text-[var(--color-on-surface-variant)] opacity-50">#{cause.rank}</span>
                    <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${cat.bg} ${cat.text}`}>{cause.category}</span>
                    <span className="text-[10px] font-mono text-[var(--color-on-surface-variant)] opacity-50">{cause.id}</span>
                  </div>
                  <p className="text-sm font-black text-[var(--color-on-surface)]">{cause.label}</p>

                  {cause.evidence.slice(0, 3).map((e, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <span className="material-symbols-outlined text-xs text-[#855300] mt-0.5 flex-shrink-0">arrow_right</span>
                      <p className="text-xs text-[var(--color-on-surface-variant)]">{e}</p>
                    </div>
                  ))}

                  {cause.reason && (
                    <div className="p-2.5 rounded-lg bg-[var(--color-surface-container-low)] border-l-2 border-[var(--color-outline-variant)]">
                      <p className="text-xs text-[var(--color-on-surface-variant)] leading-relaxed">{cause.reason}</p>
                    </div>
                  )}

                  <div className="flex items-start gap-1.5 p-2 rounded-lg bg-secondary/5 border border-secondary/20">
                    <span className="material-symbols-outlined text-xs text-secondary mt-0.5 flex-shrink-0">build</span>
                    <p className="text-xs text-secondary font-medium">{cause.remediation}</p>
                  </div>
                </div>

                {/* Right: probability */}
                <div className={`flex-shrink-0 flex flex-col items-center gap-1 px-3 py-2 rounded-xl ${probBg}`}>
                  <span className={`font-headline text-2xl font-black ${probColor}`}>{pct}%</span>
                  <span className={`text-[9px] font-black tracking-widest ${probColor}`}>{probLabel}</span>
                  <div className="w-12 h-1.5 bg-[var(--color-outline-variant)] rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${isHigh ? "bg-error" : isMedium ? "bg-[#855300]" : "bg-primary"}`} style={{ width: `${pct}%` }} />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────── Performance Tab ──────────────────────────────────
function PerformanceTab({ data }: { data: PerformanceMetrics }) {
  if (!data) return <EmptyState />;

  const metrics = [
    ["Avg Latency", `${(data.avg_latency ?? 0).toFixed(0)}ms`],
    ["P95 Latency", `${(data.p95_latency ?? 0).toFixed(0)}ms`],
    ["Throughput", `${(data.throughput ?? 0).toFixed(2)} req/s`],
    ["Error Rate", `${(data.error_rate ?? 0).toFixed(1)}%`],
    ["Min Latency", `${(data.min_latency ?? 0).toFixed(0)}ms`],
    ["Max Latency", `${(data.max_latency ?? 0).toFixed(0)}ms`],
    ["P50 (Median)", `${(data.median_latency ?? 0).toFixed(0)}ms`],
    ["P99 Latency", `${(data.p99_latency ?? 0).toFixed(0)}ms`],
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map(([label, value]) => (
          <MetricCard key={label} label={label} value={value} />
        ))}
      </div>

      <div className="p-4 rounded-xl bg-[var(--color-surface-container-low)]">
        <p className="text-xs font-bold uppercase tracking-widest text-[var(--color-on-surface-variant)] opacity-60 mb-2">Details</p>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-[var(--color-on-surface-variant)] opacity-70">Total Requests</span>
            <span className="font-semibold">{data.total_requests ?? 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[var(--color-on-surface-variant)] opacity-70">Successful</span>
            <span className="font-semibold text-secondary">{data.successful ?? 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[var(--color-on-surface-variant)] opacity-70">Errors</span>
            <span className="font-semibold text-error">{data.errors ?? 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────── Load Tab ─────────────────────────────────────────
function LoadTab({ data }: { data: LoadMetrics }) {
  if (!data) return <EmptyState />;

  const errRate = data.error_rate ?? 0;
  const errColor = errRate < 1 ? "text-secondary" : errRate < 5 ? "text-[#855300]" : "text-error";
  const assessment = errRate < 1 ? "Excellent — minimal errors" : errRate < 5 ? "Acceptable — some errors detected" : "Critical — high error rate";

  const metrics = [
    ["Concurrent Users", data.concurrent_users ?? 0],
    ["Duration", `${(data.duration_seconds ?? 0).toFixed(1)}s`],
    ["Total Requests", data.total_requests ?? 0],
    ["Successful", data.successful ?? 0],
    ["Failed", data.errors ?? 0],
    ["Avg Latency", `${(data.avg_latency ?? 0).toFixed(0)}ms`],
    ["P95 Latency", `${(data.p95_latency ?? 0).toFixed(0)}ms`],
    ["Throughput", `${(data.requests_per_second ?? 0).toFixed(2)} rps`],
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map(([label, value]) => (
          <MetricCard key={label} label={String(label)} value={String(value)} />
        ))}
      </div>

      <div className={`p-4 rounded-xl border flex items-center gap-3 ${
        errRate < 1 ? "border-secondary/30 bg-secondary/5" :
        errRate < 5 ? "border-[#855300]/30 bg-[#855300]/5" :
        "border-error/30 bg-error/5"
      }`}>
        <span className={`material-symbols-outlined ${errColor}`}>
          {errRate < 1 ? "check_circle" : errRate < 5 ? "warning" : "error"}
        </span>
        <div>
          <p className={`text-sm font-black ${errColor}`}>Error Rate: {errRate.toFixed(1)}%</p>
          <p className="text-xs text-[var(--color-on-surface-variant)] opacity-70">{assessment}</p>
        </div>
      </div>

      {data.tool_used && (
        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-50">Tool: {data.tool_used}</p>
      )}
    </div>
  );
}

// ─────────────────────── Shared mini-components ────────────────────────────
function TestResultRow({ result }: { result: TestResult }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-[var(--color-outline-variant)] first:border-0">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between p-3 hover:bg-[var(--color-surface-container-low)] transition-colors text-left gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <span className={`material-symbols-outlined text-sm flex-shrink-0 ${result.passed ? "text-secondary" : "text-error"}`}>
            {result.passed ? "check_circle" : "cancel"}
          </span>
          <p className="text-xs font-semibold truncate">{result.test_name}</p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs text-[var(--color-on-surface-variant)] opacity-60">{result.latency_ms.toFixed(0)}ms</span>
          <span className={`text-xs font-black ${result.passed ? "text-secondary" : "text-error"}`}>{(result.score * 100).toFixed(0)}%</span>
          <span className="material-symbols-outlined text-sm text-[var(--color-on-surface-variant)]">{open ? "expand_less" : "expand_more"}</span>
        </div>
      </button>
      {open && (
        <div className="px-4 pb-3 space-y-2">
          <ConversationBlock input={result.input_text} output={result.output_text} />
          {result.reasoning && (
            <p className="text-xs text-[var(--color-on-surface-variant)] italic">{result.reasoning}</p>
          )}
          {!result.passed && result.failure_reason && (
            <div className="p-3 rounded-lg border border-[#6200ee]/20 bg-[#6200ee]/5 space-y-1.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="material-symbols-outlined text-xs text-[#6200ee]">travel_explore</span>
                <p className="text-[10px] font-black uppercase tracking-wider text-[#6200ee]">Root Cause</p>
                {result.failure_taxonomy_id && result.failure_taxonomy_label && (
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#ede7f6] text-[#6200ee]">
                    {result.failure_taxonomy_id} · {result.failure_taxonomy_label}
                  </span>
                )}
              </div>
              <p className="text-xs text-[var(--color-on-surface-variant)] leading-relaxed">{result.failure_reason}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ConversationBlock({
  input, output, inputLabel = "Input", outputLabel = "Response"
}: { input: string; output: string; inputLabel?: string; outputLabel?: string }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
      <div className="p-3 rounded-lg bg-[var(--color-surface-container-low)]">
        <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-on-surface-variant)] opacity-60 mb-1">{inputLabel}</p>
        <p className="text-xs font-mono break-all">{input || "—"}</p>
      </div>
      <div className="p-3 rounded-lg bg-[var(--color-surface-container-low)]">
        <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-on-surface-variant)] opacity-60 mb-1">{outputLabel}</p>
        <p className="text-xs break-all">{output || "—"}</p>
      </div>
    </div>
  );
}

function SummaryRow4({ items }: { items: Array<{ label: string; value: number | string; color?: string }> }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {items.map((item) => (
        <div key={item.label} className="p-4 rounded-xl bg-[var(--color-surface-container-low)] text-center">
          <p className={`font-headline text-2xl font-black ${item.color || "text-[var(--color-on-surface)]"}`}>{item.value}</p>
          <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60 mt-0.5">{item.label}</p>
        </div>
      ))}
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-4 rounded-xl bg-[var(--color-surface-container-low)] text-center">
      <p className="font-headline text-xl font-black text-[var(--color-on-surface)]">{value}</p>
      <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60 mt-0.5">{label}</p>
    </div>
  );
}

function ExportCard({ icon, title, description, onClick }: { icon: string; title: string; description: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="card p-5 text-left space-y-3 hover:shadow-lg hover:shadow-primary/5 transition-shadow group"
    >
      <span className="material-symbols-outlined text-2xl text-primary group-hover:scale-110 transition-transform">{icon}</span>
      <div>
        <p className="font-headline text-sm font-black">{title}</p>
        <p className="text-xs text-[var(--color-on-surface-variant)] opacity-60 mt-1">{description}</p>
      </div>
      <div className="flex items-center gap-2 text-xs font-semibold text-primary">
        <span className="material-symbols-outlined text-sm">download</span>
        Download
      </div>
    </button>
  );
}

function EmptyState() {
  return (
    <div className="py-12 text-center space-y-2">
      <span className="material-symbols-outlined text-3xl text-[var(--color-on-surface-variant)] opacity-30">data_usage</span>
      <p className="text-sm text-[var(--color-on-surface-variant)] opacity-50">No data available for this phase.</p>
    </div>
  );
}
