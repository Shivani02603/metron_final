"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface Project {
  id: string;
  name: string;
  endpoint: string;
  status: string;
  runs: number;
  lastScore: number | string;
  type: string;
}

export default function ProjectHub() {
  const router = useRouter();
  const [showModal, setShowModal] = useState(false);

  // Form state — basics only (test config lives in splitter/builder)
  const [projectName, setProjectName] = useState("");
  const [projectEndpoint, setProjectEndpoint] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

  const [projects, setProjects] = useState<Project[]>([]);

  // Load persisted projects from API on mount
  useEffect(() => {
    fetch("/api/projects", { credentials: "include" })
      .then((r) => r.json())
      .then((data) => {
        if (!data.projects) return;
        setProjects(
          data.projects.map((p: { project_id: string; name: string; endpoint: string }) => ({
            id: p.project_id,
            name: p.name || p.endpoint,
            endpoint: p.endpoint,
            status: "Ready",
            runs: 0,
            lastScore: "---",
            type: "AI System",
          }))
        );
      })
      .catch(() => {});
  }, []);

  const handleProjectClick = (projectId: string) => {
    router.push(`/dashboard/project/${projectId}`);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setUploadedFile(file);
  };

  const handleConnect = () => {
    if (!projectEndpoint || !uploadedFile) return;
    setIsSubmitting(true);
    setSubmitError("");

    const reader = new FileReader();
    reader.onload = async () => {
      const tempId = Date.now().toString();
      const projectData = {
        name: projectName || projectEndpoint,
        endpoint: projectEndpoint,
        apiKey: apiKey,
        documentText: reader.result as string,
        documentName: uploadedFile!.name,
      };

      // Write to sessionStorage for current session sub-pages
      sessionStorage.setItem(`project_${tempId}`, JSON.stringify(projectData));

      // Persist to server so it survives logout/login
      try {
        await fetch("/api/projects", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: tempId,
            name: projectData.name,
            endpoint: projectData.endpoint,
            api_key: projectData.apiKey,
            document_text: projectData.documentText,
            document_name: projectData.documentName,
          }),
        });
      } catch {
        // Non-fatal — session still works
      }

      setProjects((prev) => [
        ...prev,
        {
          id: tempId,
          name: projectData.name,
          endpoint: projectData.endpoint,
          status: "Ready",
          runs: 0,
          lastScore: "---",
          type: "AI System",
        },
      ]);

      setIsSubmitting(false);
      setShowModal(false);
      setProjectName("");
      setProjectEndpoint("");
      setApiKey("");
      setUploadedFile(null);

      router.push(`/dashboard/project/${tempId}`);
    };

    reader.onerror = () => {
      setSubmitError("Could not read file. Try a .txt or .md file.");
      setIsSubmitting(false);
    };

    reader.readAsText(uploadedFile);
  };

  const closeModal = () => {
    if (!isSubmitting) {
      setShowModal(false);
      setSubmitError("");
    }
  };

  return (
    <>
    <div className="space-y-10 animate-fade-in max-w-7xl mx-auto pb-20">
      {/* ─── Page Header ─────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-end justify-between items-start gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-primary" />
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">Active Workspace</span>
          </div>
          <h1 className="font-headline text-5xl font-black text-[var(--color-on-surface)] tracking-tighter">Project Hub</h1>
          <p className="text-[var(--color-on-surface-variant)] text-sm font-medium opacity-60">Connect an AI system and run evaluations.</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="group flex items-center gap-3 px-8 py-4 rounded-2xl btn-primary text-sm shadow-xl shadow-[#00668a]/10 hover:shadow-2xl hover:shadow-[#00668a]/15 transition-all"
        >
          <span className="material-symbols-outlined text-xl leading-none group-hover:rotate-90 transition-transform duration-500">add</span>
          Connect New Project
        </button>
      </div>

      {/* ─── Projects Grid ───────────────────────────────────── */}
      {projects.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
          <div className="w-16 h-16 rounded-2xl bg-[var(--color-surface-container-low)] border border-[var(--color-outline-variant)] flex items-center justify-center">
            <span className="material-symbols-outlined text-3xl text-[var(--color-outline)]">hub</span>
          </div>
          <div className="space-y-1">
            <p className="font-headline text-lg font-black text-[var(--color-on-surface)]">No projects yet</p>
            <p className="text-sm text-[var(--color-on-surface-variant)] opacity-60">Click &ldquo;Connect New Project&rdquo; to add your first AI system.</p>
          </div>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {projects.map((proj) => (
          <div
            key={proj.id}
            onClick={() => handleProjectClick(proj.id)}
            className="group relative p-8 rounded-[2.5rem] bg-[var(--color-surface-container-lowest)] border border-[var(--color-outline-variant)] border-opacity-20 shadow-card hover:shadow-card-lg transition-all cursor-pointer overflow-hidden animate-fade-in"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-primary/0 to-primary/3 opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="relative z-10 space-y-5">
              <div className="flex items-center justify-between">
                <div className="w-14 h-14 rounded-2xl bg-[var(--color-surface-container-low)] flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-white transition-all duration-500 shadow-sm border border-black/5">
                  <span className="material-symbols-outlined text-3xl">
                    {proj.type === "RAG System" ? "database" : proj.type === "Chatbot" ? "forum" : "hub"}
                  </span>
                </div>
                <div className="px-3 py-1.5 rounded-full bg-[#6bff8f]/10 border border-[#6bff8f]/20 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#006e2f] pulse-orb" />
                  <span className="text-[9px] font-black uppercase tracking-widest text-[#006e2f]">{proj.status}</span>
                </div>
              </div>
              <div className="space-y-1.5">
                <h3 className="font-headline text-2xl font-black text-[var(--color-on-surface)] tracking-tight truncate leading-none group-hover:text-primary transition-colors">{proj.name}</h3>
                <p className="text-[10px] font-bold text-[var(--color-outline)] truncate flex items-center gap-1.5 opacity-60 italic">
                  <span className="material-symbols-outlined text-xs">link</span> {proj.endpoint}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-8 py-1">
                <div className="space-y-0.5">
                  <p className="text-[9px] font-black text-[var(--color-outline)] uppercase tracking-widest leading-none">Compliance</p>
                  <p className="text-3xl font-black text-primary font-headline tracking-tighter mt-1">{proj.lastScore}{typeof proj.lastScore === "number" ? "%" : ""}</p>
                </div>
                <div className="space-y-0.5 text-right border-l border-[var(--color-outline-variant)] border-opacity-10 pl-8">
                  <p className="text-[9px] font-black text-[var(--color-outline)] uppercase tracking-widest leading-none">Runs</p>
                  <p className="text-3xl font-black text-[var(--color-on-surface)] font-headline tracking-tighter mt-1">{proj.runs}</p>
                </div>
              </div>
              <div className="pt-6 flex items-center justify-between border-t border-[var(--color-outline-variant)] border-opacity-10">
                <span className="px-2.5 py-1 rounded-lg bg-[var(--color-surface-container-low)] text-[8px] font-black uppercase tracking-widest text-[var(--color-outline)] border border-outline-variant/10">{proj.type}</span>
                <span className="material-symbols-outlined text-[var(--color-outline)] group-hover:translate-x-1 group-hover:text-primary transition-all duration-300">arrow_forward</span>
              </div>
            </div>
          </div>
        ))}
      </div>

    </div>

      {/* ─── CONNECT PROJECT MODAL ────────────────────────────── */}
      {/* Rendered outside the animated wrapper so fixed positioning covers the full viewport */}
      {showModal && (
        <div className="fixed inset-0 z-[1000] flex items-center justify-center p-4">
          {/* Full-screen backdrop */}
          <div className="fixed inset-0 bg-[#001e2c]/60 backdrop-blur-md" onClick={closeModal} />

          {/* Modal card — max-h so button is always visible; form area scrolls */}
          <div className="relative w-full max-w-lg bg-[var(--color-surface-container-lowest)] rounded-[2.5rem] shadow-[0px_64px_128px_-12px_rgba(0,30,44,0.4)] flex flex-col animate-fade-in border border-white/5 max-h-[90vh]">

            {/* Header — fixed at top */}
            <div className="p-8 pb-4 flex justify-between items-start border-b border-[var(--color-outline-variant)] border-opacity-10 flex-shrink-0">
              <div className="space-y-1">
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-primary/10 border border-primary/20">
                  <span className="material-symbols-outlined text-primary text-sm leading-none">smart_toy</span>
                  <span className="text-[8px] font-black uppercase tracking-widest text-primary">Node Setup</span>
                </div>
                <h2 className="font-headline text-3xl font-black text-[var(--color-on-surface)] tracking-tighter">Connect Project</h2>
                <p className="text-[10px] text-outline opacity-50">You'll choose test configuration on the next screen.</p>
              </div>
              <button onClick={closeModal} className="w-10 h-10 rounded-xl bg-surface-container-low hover:bg-error/10 hover:text-error transition-all flex items-center justify-center text-outline flex-shrink-0">
                <span className="material-symbols-outlined text-xl">close</span>
              </button>
            </div>

            {/* Form — scrollable */}
            <div className="p-8 space-y-6 overflow-y-auto flex-1">

              {/* Project Name */}
              <div className="space-y-2">
                <label className="text-[10px] font-black uppercase tracking-[0.25em] text-[var(--color-outline)]">Project Name (optional)</label>
                <input
                  type="text"
                  placeholder="e.g. Wealth Management Chatbot"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  className="w-full px-6 py-4 rounded-2xl bg-[var(--color-surface-container-low)] border-none ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-[var(--color-primary)] transition-all outline-none font-headline font-bold text-lg text-on-surface"
                />
              </div>

              {/* Target Endpoint */}
              <div className="space-y-2">
                <label className="text-[10px] font-black uppercase tracking-[0.25em] text-[var(--color-outline)]">Target API Endpoint <span className="text-red-400">*</span></label>
                <div className="relative group">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline/50 group-focus-within:text-primary transition-colors text-xl">link</span>
                  <input
                    type="url"
                    required
                    placeholder="https://your-ai-system.com/api/chat"
                    value={projectEndpoint}
                    onChange={(e) => setProjectEndpoint(e.target.value)}
                    className="w-full pl-12 pr-6 py-4 rounded-2xl bg-[var(--color-surface-container-low)] border-none ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-[var(--color-primary)] transition-all outline-none font-bold text-sm text-on-surface"
                  />
                </div>
              </div>

              {/* API Key */}
              <div className="space-y-2">
                <label className="text-[10px] font-black uppercase tracking-[0.25em] text-[var(--color-outline)]">API Key / Bearer Token (optional)</label>
                <div className="relative group">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline/50 group-focus-within:text-primary transition-colors text-xl">key</span>
                  <input
                    type="password"
                    placeholder="sk-... or Bearer token"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="w-full pl-12 pr-6 py-4 rounded-2xl bg-[var(--color-surface-container-low)] border-none ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-[var(--color-primary)] transition-all outline-none font-bold text-sm text-on-surface"
                  />
                </div>
              </div>

              {/* Document Upload */}
              <div className="space-y-3">
                <label className="text-[10px] font-black uppercase tracking-[0.25em] text-[var(--color-outline)]">System Document <span className="text-red-400">*</span></label>
                <div className="relative group p-8 border-2 border-dashed border-[var(--color-outline-variant)] border-opacity-30 rounded-[2rem] bg-[var(--color-surface-container-low)]/50 hover:bg-primary/5 hover:border-primary/50 transition-all text-center space-y-3 cursor-pointer overflow-hidden">
                  <input
                    type="file"
                    className="absolute inset-0 opacity-0 cursor-pointer z-20"
                    onChange={handleFileUpload}
                    accept=".txt,.md,.json,.pdf,.docx"
                  />
                  <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-2 transition-all duration-500 shadow-sm ${uploadedFile ? "bg-[#6bff8f] text-[#006e2f] scale-110" : "bg-primary/5 text-primary group-hover:bg-primary group-hover:text-white"}`}>
                    <span className="material-symbols-outlined text-3xl">{uploadedFile ? "task_alt" : "quick_reference_all"}</span>
                  </div>
                  {uploadedFile ? (
                    <div className="animate-fade-in">
                      <p className="font-headline font-black text-on-surface text-base truncate px-4">{uploadedFile.name}</p>
                      <p className="text-[10px] font-bold text-[#006e2f] uppercase tracking-widest mt-1">Ready</p>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <h4 className="font-headline font-black text-on-surface">Upload Knowledge Document</h4>
                      <p className="text-[9px] text-[var(--color-outline)] font-bold uppercase tracking-widest opacity-50">txt · md · json (describes your AI system)</p>
                    </div>
                  )}
                </div>
              </div>

              {submitError && (
                <div className="p-4 rounded-2xl bg-error/10 border border-error/30 text-error text-xs font-bold">{submitError}</div>
              )}
            </div>

            {/* Footer — fixed at bottom, always visible */}
            <div className="px-8 pb-8 pt-4 flex-shrink-0">
              <button
                onClick={handleConnect}
                disabled={isSubmitting || !projectEndpoint || !uploadedFile}
                className="w-full py-5 rounded-2xl btn-primary shadow-2xl shadow-[#00668a]/20 disabled:opacity-30 disabled:cursor-not-allowed group"
              >
                {isSubmitting ? (
                  <div className="flex items-center justify-center gap-3">
                    <span className="material-symbols-outlined animate-spin text-xl">progress_activity</span>
                    <span className="font-black uppercase tracking-widest text-sm">Reading document...</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-3">
                    <span className="font-headline font-black text-lg uppercase">Connect & Select Strategy</span>
                    <span className="material-symbols-outlined group-hover:translate-x-2 transition-transform">arrow_forward</span>
                  </div>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
