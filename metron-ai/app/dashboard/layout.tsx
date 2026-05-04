"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { fetchAuthSession, signOut } from "aws-amplify/auth";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [userEmail, setUserEmail] = useState("");

  console.log("[DashboardLayout] Rendering (not just mounted)", {
    pathname,
    timestamp: new Date().toISOString()
  });

  useEffect(() => {
    console.log("[DashboardLayout] ✓ Component mounted at", new Date().toISOString(), "pathname:", pathname);
    let isMounted = true;

    fetchAuthSession()
      .then((session) => {
        if (!isMounted) return;
        console.log("[DashboardLayout] First fetchAuthSession result:", {
          hasTokens: !!session.tokens,
          tokenKeys: session.tokens ? Object.keys(session.tokens) : [],
          hasIdToken: !!session.tokens?.idToken,
          hasAccessToken: !!session.tokens?.accessToken,
          idTokenPayload: session.tokens?.idToken?.payload,
          credentialsSource: (session as any)?.credentials?.source,
        });

        if (session.tokens) {
          const email = (session.tokens.idToken?.payload?.email as string) ?? "";
          console.log("[DashboardLayout] Got email from token on first try:", email);
          setUserEmail(email);
          return;
        }

        // tokens absent — one retry after a tick (handles Amplify init race)
        console.log("[DashboardLayout] No tokens on first attempt, waiting 300ms and retrying...");
        return new Promise<void>((resolve) => setTimeout(resolve, 300))
          .then(() => {
            if (!isMounted) return;
            console.log("[DashboardLayout] Retrying fetchAuthSession...");
            return fetchAuthSession();
          })
          .then((s) => {
            if (!isMounted) return;
            console.log("[DashboardLayout] Second fetchAuthSession result:", {
              hasTokens: !!s.tokens,
              tokenKeys: s.tokens ? Object.keys(s.tokens) : [],
              hasIdToken: !!s.tokens?.idToken,
            });

            if (s.tokens) {
              const email = (s.tokens.idToken?.payload?.email as string) ?? "";
              console.log("[DashboardLayout] Got email from token on retry:", email);
              setUserEmail(email);
            } else {
              console.warn("[DashboardLayout] Still no tokens after retry, but allowing dashboard to load");
            }
          });
      })
      .catch((err) => {
        if (!isMounted) return;
        console.error("[DashboardLayout] Error fetching auth session:", {
          message: err?.message,
          code: (err as any)?.code,
          name: err?.name,
          fullError: err
        });
        console.warn("[DashboardLayout] Auth session fetch failed, but allowing dashboard to load (API calls will handle auth)");
      });

    return () => {
      isMounted = false;
    };
  }, [router]);

  const handleLogout = async () => {
    await signOut();
    window.location.href = "/";
  };

  const navLinks = [
    { name: "Project Hub", icon: "grid_view", href: "/dashboard", active: true },
    { name: "Recent Runs", icon: "history", href: "#", active: false, badge: "Soon" },
    { name: "Team Settings", icon: "group", href: "#", active: false, badge: "Soon" },
  ];

  return (
    <div className="flex h-screen bg-[var(--color-background)] overflow-hidden">
      {/* ─── Main Global Sidebar (with Collapse Logic) ───────────── */}
      <aside 
        className={`flex flex-col bg-[var(--color-surface-container-lowest)] border-r border-[var(--color-outline-variant)] border-opacity-30 relative z-[100] transition-all duration-500 ease-in-out ${
          isCollapsed ? "w-20" : "w-64"
        }`}
      >
        {/* Toggle Button (The Closing Arrow) */}
        <button 
          type="button"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="absolute -right-3 top-20 w-6 h-6 rounded-full bg-white border border-outline-variant/30 flex items-center justify-center text-primary shadow-sm hover:shadow-md transition-all z-[110]"
        >
          <span className={`material-symbols-outlined text-sm transition-transform duration-500 ${isCollapsed ? 'rotate-180' : ''}`}>
             chevron_left
          </span>
        </button>

        <div className={`p-6 flex items-center gap-3 transition-all duration-300 ${isCollapsed ? 'justify-center px-0' : ''}`}>
          <div className="w-9 h-9 min-w-[36px] rounded-xl bg-[var(--color-primary)] flex items-center justify-center shadow-lg shadow-[#00668a]/20">
            <span className="material-symbols-outlined text-white text-xl font-bold italic">M</span>
          </div>
          {!isCollapsed && (
             <span className="font-headline text-xl font-black text-[var(--color-primary)] tracking-tighter animate-fade-in">MetronAI</span>
          )}
        </div>

        <nav className="flex-1 px-4 py-4 space-y-2">
          {navLinks.map((link) => {
            const isActive = pathname === link.href && link.active;
            return (
              <Link
                key={link.name}
                href={link.href}
                onClick={(e) => { if (!link.active) e.preventDefault(); }}
                className={`flex items-center gap-3 px-4 py-3.5 rounded-xl text-sm font-bold transition-all group relative ${
                  isCollapsed ? 'justify-center px-0' : ''
                } ${
                  isActive
                    ? "bg-[var(--color-primary)] text-white shadow-lg shadow-[#00668a]/20"
                    : link.active 
                      ? "text-[var(--color-on-surface-variant)] hover:bg-[var(--color-surface-container-low)]"
                      : "text-[var(--color-outline)] opacity-40 cursor-not-allowed"
                }`}
              >
                <span className="material-symbols-outlined text-xl">{link.icon}</span>
                {!isCollapsed && (
                  <span className="animate-fade-in flex-1">{link.name}</span>
                )}
                {!isCollapsed && link.badge && (
                  <span className="text-[8px] px-1.5 py-0.5 rounded bg-[var(--color-outline-variant)]/20 uppercase tracking-tighter">{link.badge}</span>
                )}
              </Link>
            );
          })}
        </nav>

        {/* User Card */}
        <div className={`p-4 border-t border-[var(--color-outline-variant)] border-opacity-10 ${isCollapsed ? 'px-2' : ''}`}>
          <div className={`flex items-center gap-3 p-2.5 rounded-2xl bg-[var(--color-surface-container-low)] transition-all ${isCollapsed ? 'justify-center rounded-xl px-0' : ''}`}>
            <div className="w-10 h-10 min-w-[40px] rounded-full bg-gradient-to-br from-[#00668a] to-[#38bdf8] flex items-center justify-center text-white text-xs font-bold ring-2 ring-white shadow-sm">
              {userEmail ? userEmail[0].toUpperCase() : "?"}
            </div>
            {!isCollapsed && (
              <div className="flex-1 overflow-hidden animate-fade-in">
                <p className="text-xs font-black text-[var(--color-on-surface)] truncate">{userEmail}</p>
              </div>
            )}
            {!isCollapsed && (
              <button type="button" onClick={handleLogout} className="w-8 h-8 rounded-lg flex items-center justify-center text-[var(--color-outline)] hover:text-red-500 hover:bg-red-50 transition-all">
                <span className="material-symbols-outlined text-xl">logout</span>
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* ─── Main View Zone ───────────────────────────── */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {/* Top Navbar */}
        <header className="h-16 flex items-center justify-between px-8 bg-white border-b border-[var(--color-outline-variant)] border-opacity-20 relative z-[40]">
           <div className="flex items-center gap-2">
              <span className="text-[9px] font-black text-outline uppercase tracking-[0.2em] opacity-40">Workspace</span>
              <span className="material-symbols-outlined text-sm text-outline opacity-20">chevron_right</span>
              <span className="text-[9px] font-black text-on-surface uppercase tracking-[0.2em]">Live Intelligence Node</span>
           </div>
           
           <div className="flex items-center gap-4">
              <div className="lg:flex hidden h-9 items-center gap-2 px-4 rounded-full bg-[var(--color-surface-container-low)] border border-outline-variant/10">
                 <span className="w-1.5 h-1.5 rounded-full bg-[#6bff8f] animate-pulse" />
                 <span className="text-[9px] font-black text-on-surface uppercase tracking-widest opacity-70">Backend Synced</span>
              </div>
           </div>
        </header>

        {/* Content Zone */}
        <div className="flex-1 overflow-y-auto p-8 animate-fade-in no-scrollbar bg-[var(--color-background)]">
           {children}
        </div>
      </main>
    </div>
  );
}
