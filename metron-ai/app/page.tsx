"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { signIn, signOut, fetchAuthSession } from "aws-amplify/auth";

console.log("[LoginPage] Component file loaded");

export default function LoginPage() {
  console.log("[LoginPage] Rendering login page component");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  console.log("[LoginPage] useRouter hook initialized");

  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      console.error("[LoginPage] Global error:", event.error);
    };
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error("[LoginPage] Unhandled rejection:", event.reason);
    };

    window.addEventListener("error", handleError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);

    return () => {
      window.removeEventListener("error", handleError);
      window.removeEventListener("unhandledrejection", handleUnhandledRejection);
    };
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      console.log("[Login] Starting sign-in for email:", email.trim());

      // Clear any stale Amplify session before attempting fresh sign-in
      try {
        console.log("[Login] Clearing stale session...");
        await signOut();
        console.log("[Login] Stale session cleared");
      } catch (err) {
        console.log("[Login] No stale session to clear:", err);
      }

      console.log("[Login] Calling signIn...");
      const output = await signIn({ username: email.trim(), password });
      console.log("[Login] SignIn output:", {
        isSignedIn: output.isSignedIn,
        nextStep: output.nextStep,
        fullOutput: JSON.stringify(output),
      });

      if (output.isSignedIn) {
        console.log("[Login] Sign-in successful, checking localStorage for tokens...");
        const storageKeys = Object.keys(localStorage);
        console.log("[Login] localStorage keys count:", storageKeys.length);

        // Verify session is accessible
        try {
          const session = await fetchAuthSession();
          console.log("[Login] Session check after signIn:", {
            hasTokens: !!session.tokens,
            hasIdToken: !!session.tokens?.idToken,
            email: session.tokens?.idToken?.payload?.email
          });
        } catch (e) {
          console.error("[Login] Failed to fetch session after signIn:", e);
        }

        setLoading(false);
        console.log("[Login] Tokens confirmed, attempting navigation to /dashboard...");

        // Use window.location.href directly since router.push() appears to be non-functional
        console.log("[Login] Using window.location.href = '/dashboard'");
        window.location.href = "/dashboard";

        // Backup: if window.location.href doesn't work, try router.push after delay
        setTimeout(() => {
          if (window.location.pathname === "/") {
            console.warn("[Login] window.location.href navigation failed, trying router.push as backup...");
            try {
              router.push("/dashboard");
            } catch (err) {
              console.error("[Login] router.push backup also failed:", err);
            }
          }
        }, 500);
      } else {
        const msg = `Sign-in incomplete (${output.nextStep?.signInStep ?? "unknown step"}). Check your email for a confirmation code.`;
        console.log("[Login] Sign-in incomplete:", msg);
        setError(msg);
        setLoading(false);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Invalid email or password";
      console.error("[Login] Error during sign-in:", { msg, fullError: err });
      setError(msg);
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--color-background)]">
      {/* ─── Left Brand Panel ─────────────────────────────── */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-16 relative overflow-hidden bg-gradient-to-br from-[#001e2c] via-[#004c69] to-[#00668a]">
        <div className="absolute -top-32 -left-32 w-[500px] h-[500px] rounded-full bg-white/5 blur-3xl" />
        <div className="absolute bottom-0 right-0 w-[600px] h-[600px] rounded-full bg-[var(--color-primary-container)] opacity-10 blur-3xl" />

        {/* Brand Logo */}
        <div className="relative z-10 flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-white/10 backdrop-blur-md flex items-center justify-center border border-white/20">
            <span className="material-symbols-outlined text-white text-2xl">monitor_heart</span>
          </div>
          <span className="text-white font-headline text-2xl font-black tracking-tighter">MetronAI</span>
        </div>

        {/* Hero Section */}
        <div className="relative z-10 space-y-8 max-w-lg">
          <div className="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/20">
            <span className="w-2.5 h-2.5 rounded-full bg-[#6bff8f] pulse-orb" />
            <span className="text-white/90 text-[10px] font-black uppercase tracking-[0.2em]">Enterprise AI QA Intelligence</span>
          </div>
          
          <div className="space-y-4">
            <h1 className="font-headline text-5xl xl:text-6xl font-extrabold text-white leading-[1.1] tracking-tight">
              Test smarter.<br />
              Ship with<br />
              <span className="text-[var(--color-primary-container)]">confidence.</span>
            </h1>
            <p className="text-white/60 text-lg leading-relaxed font-medium">
              Validate Functional, Security, Performance, and Load stability for your AI agents and RAG systems.
            </p>
          </div>
        </div>

        {/* Footer Meta */}
        <div className="relative z-10 flex items-center gap-8 pt-8 border-t border-white/5">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-white/40 text-lg">verified_user</span>
            <span className="text-white/40 text-[10px] font-bold uppercase tracking-widest leading-none">SOC 2 Compliant</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-white/40 text-lg">lock</span>
            <span className="text-white/40 text-[10px] font-bold uppercase tracking-widest leading-none">Enterprise SSO</span>
          </div>
        </div>
      </div>

      {/* ─── Right Login Area ──────────────────────────────── */}
      <div className="flex-1 flex flex-col items-center justify-center p-8 lg:p-16 bg-[var(--color-background)] animate-fade-in">
        <div className="w-full max-w-[420px] space-y-10">
          {/* Mobile Logo Only */}
          <div className="lg:hidden flex items-center gap-3 mb-10">
            <div className="w-11 h-11 rounded-xl bg-[var(--color-primary)] flex items-center justify-center shadow-lg shadow-primary/20">
              <span className="material-symbols-outlined text-white">monitor_heart</span>
            </div>
            <span className="font-headline text-2xl font-black text-[var(--color-primary)] tracking-tighter">MetronAI</span>
          </div>

          <div className="space-y-2">
            <h2 className="font-headline text-3xl font-extrabold text-[var(--color-on-surface)] tracking-tight">Welcome back</h2>
            <p className="text-[var(--color-on-surface-variant)] font-medium">Please sign in to your workspace.</p>
          </div>

          {/* Form */}
          <form className="space-y-6" autoComplete="on" onSubmit={handleLogin}>
            <div className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-[10px] font-black uppercase tracking-[0.1em] text-[var(--color-on-surface-variant)]">Work Email</label>
                <div className="relative group">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-[var(--color-outline)] group-focus-within:text-[var(--color-primary)] transition-colors">mail</span>
                  <input
                    type="email"
                    placeholder="name@company.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full pl-12 pr-4 py-4 rounded-xl bg-[var(--color-surface-container-low)] border-none ring-1 ring-[#00668a]/10 focus:ring-2 focus:ring-[var(--color-primary)] transition-all outline-none"
                    required
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-[10px] font-black uppercase tracking-[0.1em] text-[var(--color-on-surface-variant)]">Password</label>
                <div className="relative group">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-[var(--color-outline)] group-focus-within:text-[var(--color-primary)] transition-colors">lock</span>
                  <input
                    type={showPassword ? "text" : "password"}
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full pl-12 pr-12 py-4 rounded-xl bg-[var(--color-surface-container-low)] border-none ring-1 ring-[#00668a]/10 focus:ring-2 focus:ring-[var(--color-primary)] transition-all outline-none"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-[var(--color-outline)] hover:text-[var(--color-primary)] transition-colors"
                  >
                    <span className="material-symbols-outlined text-xl">{showPassword ? "visibility_off" : "visibility"}</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-1">
              <a href="#" className="text-xs font-bold text-[var(--color-primary)] hover:text-[var(--color-primary-container)] transition-colors">Forgot password?</a>
            </div>

            {error && (
              <p className="text-sm font-semibold text-red-500 bg-red-50 px-4 py-3 rounded-xl">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-4 rounded-xl btn-primary shadow-xl shadow-[#00668a]/10 flex items-center justify-center gap-3 disabled:opacity-50"
            >
              {loading ? (
                <span className="material-symbols-outlined animate-spin">progress_activity</span>
              ) : (
                <>Sign In to MetronAI <span className="material-symbols-outlined">arrow_forward</span></>
              )}
            </button>
          </form>

          {/* Social Sign In */}
          <div className="flex items-center gap-4 py-2">
            <div className="h-px flex-1 bg-[var(--color-outline-variant)] opacity-30" />
            <span className="text-[10px] font-black uppercase tracking-widest text-[var(--color-outline)]">or continue with</span>
            <div className="h-px flex-1 bg-[var(--color-outline-variant)] opacity-30" />
          </div>

          <button type="button" className="w-full py-4 rounded-xl border border-[var(--color-outline-variant)] border-opacity-40 flex items-center justify-center gap-3 text-sm font-bold text-[var(--color-on-surface)] hover:bg-[var(--color-surface-container-low)] transition-all">
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
            </svg>
            Google Workspace
          </button>

          <footer className="pt-8 text-center">
            <p className="text-xs text-[var(--color-on-surface-variant)] font-medium">
              Don&apos;t have an account?{" "}
              <a href="/register" className="text-[var(--color-primary)] font-bold hover:underline">
                Create one
              </a>
            </p>
          </footer>
        </div>
      </div>
    </div>
  );
}
