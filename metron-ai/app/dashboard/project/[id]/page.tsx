"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { authFetch } from "@/lib/api";

interface Run {
  run_id: string;
  timestamp: string;
  health_score: number | null;
  domain: string;
  application_type: string;
  status: string;
}

interface Project {
  project_id: string;
  name: string;
  endpoint: string;
  document_name: string;
}

export default function ProjectLanding() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [project, setProject] = useState<Project | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!projectId) return;

    Promise.all([
      authFetch(`/api/projects/${projectId}`).then((r) => r.ok ? r.json() : null),
      authFetch(`/api/project/${projectId}/runs`).then((r) => r.ok ? r.json() : { runs: [] }),
    ])
      .then(([proj, runsData]) => {
        if (!proj) {
          setError("Project not found.");
          setLoading(false);
          return;
        }
        setProject(proj);
        setRuns(runsData.runs || []);

        // Re-populate sessionStorage so configure/builder pages work
        sessionStorage.setItem(
          `project_${projectId}`,
          JSON.stringify({
            name: proj.name,
            endpoint: proj.endpoint,
            apiKey: proj.api_key || "",
            documentText: proj.document_text || "",
            documentName: proj.document_name || "",
          })
        );
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to load project.");
        setLoading(false);
      });
  }, [projectId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <span className="material-symbols-outlined text-4xl text-primary animate-spin">
          progress_activity
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 text-red-500 font-bold">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-10 animate-fade-in max-w-5xl mx-auto pb-20">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-1.5">
          <button
            onClick={() => router.push("/dashboard")}
            className="flex items-center gap-1.5 text-[10px] font-black uppercase tracking-widest text-[var(--color-outline)] hover:text-primary transition-colors mb-2"
          >
            <span className="material-symbols-outlined text-sm">arrow_back</span>
            Project Hub
          </button>
          <h1 className="font-headline text-4xl font-black text-[var(--color-on-surface)] tracking-tighter">
            {project?.name}
          </h1>
          <p className="text-[10px] font-bold text-[var(--color-outline)] flex items-center gap-1.5 opacity-60 italic">
            <span className="material-symbols-outlined text-xs">link</span>
            {project?.endpoint}
          </p>
        </div>
        <button
          onClick={() => router.push(`/dashboard/project/${projectId}/configure`)}
          className="group flex items-center gap-3 px-8 py-4 rounded-2xl btn-primary text-sm shadow-xl shadow-[#00668a]/10 hover:shadow-2xl transition-all"
        >
          <span className="material-symbols-outlined text-xl leading-none group-hover:rotate-90 transition-transform duration-500">
            add
          </span>
          Start New Test Run
        </button>
      </div>

      {/* Run History */}
      <div className="space-y-4">
        <h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-[var(--color-outline)]">
          Run History
        </h2>

        {runs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 gap-4 text-center rounded-[2rem] bg-[var(--color-surface-container-lowest)] border border-[var(--color-outline-variant)] border-opacity-20">
            <span className="material-symbols-outlined text-4xl text-[var(--color-outline)]">
              history
            </span>
            <p className="font-headline text-lg font-black text-[var(--color-on-surface)]">
              No runs yet
            </p>
            <p className="text-sm text-[var(--color-on-surface-variant)] opacity-60">
              Click &ldquo;Start New Test Run&rdquo; to evaluate this AI system.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {runs.map((run, i) => {
              const score = run.health_score != null ? Math.round(run.health_score * 100) : null;
              const date = new Date(run.timestamp + "Z").toLocaleString();
              return (
                <div
                  key={run.run_id}
                  className="flex items-center justify-between p-6 rounded-2xl bg-[var(--color-surface-container-lowest)] border border-[var(--color-outline-variant)] border-opacity-20 hover:border-primary/30 transition-all group"
                >
                  <div className="flex items-center gap-6">
                    <div className="w-10 h-10 rounded-xl bg-[var(--color-surface-container-low)] flex items-center justify-center text-[var(--color-outline)] font-black font-headline text-sm">
                      #{runs.length - i}
                    </div>
                    <div className="space-y-0.5">
                      <p className="text-sm font-black text-[var(--color-on-surface)]">{date}</p>
                      <p className="text-[10px] font-bold text-[var(--color-outline)] uppercase tracking-widest opacity-60">
                        {run.domain || "General"} · {run.application_type || "AI System"}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-8">
                    <div className="text-right">
                      <p className="text-[9px] font-black text-[var(--color-outline)] uppercase tracking-widest">
                        Health Score
                      </p>
                      <p
                        className={`text-2xl font-black font-headline tracking-tighter ${
                          score == null
                            ? "text-[var(--color-outline)]"
                            : score >= 75
                            ? "text-[#006e2f]"
                            : score >= 50
                            ? "text-amber-600"
                            : "text-red-600"
                        }`}
                      >
                        {score != null ? `${score}%` : "---"}
                      </p>
                    </div>
                    <button
                      onClick={() =>
                        router.push(
                          `/dashboard/project/${projectId}/results?run=${run.run_id}`
                        )
                      }
                      className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary/10 text-primary font-black text-xs uppercase tracking-widest hover:bg-primary hover:text-white transition-all"
                    >
                      View Report
                      <span className="material-symbols-outlined text-sm">arrow_forward</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
