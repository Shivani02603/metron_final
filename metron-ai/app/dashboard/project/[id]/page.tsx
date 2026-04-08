"use client";

import { useEffect } from "react";
import { useParams, useRouter } from "next/navigation";

export default function ProjectRedirect() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  useEffect(() => {
    if (projectId) {
      router.replace(`/dashboard/project/${projectId}/configure`);
    }
  }, [projectId, router]);

  return (
    <div className="flex items-center justify-center h-64">
      <div className="flex flex-col items-center gap-4">
        <span className="material-symbols-outlined text-4xl text-primary animate-spin">progress_activity</span>
        <p className="text-sm text-[var(--color-on-surface-variant)] opacity-60">Loading project…</p>
      </div>
    </div>
  );
}
