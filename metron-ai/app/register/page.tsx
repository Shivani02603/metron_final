"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";

// Role types for display
const ROLE_LABELS: Record<string, { label: string; color: string; icon: string }> = {
  admin: { label: "Admin", color: "bg-primary/10 text-primary", icon: "admin_panel_settings" },
  qa_lead: { label: "QA Lead", color: "bg-secondary/10 text-secondary", icon: "manage_accounts" },
  qa_viewer: { label: "QA Viewer", color: "bg-tertiary/10 text-tertiary", icon: "visibility" },
};

function RegisterForm() {
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const roleParam = searchParams.get("role") || "qa_viewer";
  const emailParam = searchParams.get("email") || "";
  const invitedBy = searchParams.get("invitedBy") || "Your Admin";

  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tokenValid, setTokenValid] = useState(true);

  const roleInfo = ROLE_LABELS[roleParam] ?? ROLE_LABELS.qa_viewer;

  useEffect(() => {
    // In production: validate the token with the backend
    if (!token) setTokenValid(false);
  }, [token]);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!name.trim()) { setError("Please enter your full name."); return; }
    if (password.length < 8) { setError("Password must be at least 8 characters."); return; }
    if (password !== confirmPassword) { setError("Passwords do not match."); return; }

    setLoading(true);
    // TODO: POST to /api/auth/register with { token, name, password, email: emailParam }
    setTimeout(() => {
      setLoading(false);
      window.location.href = "/dashboard";
    }, 1500);
  };

  // Invalid or missing token
  if (!tokenValid) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background px-6">
        <div className="text-center max-w-sm animate-fade-in">
          <div className="w-16 h-16 rounded-2xl bg-error-container flex items-center justify-center mx-auto mb-6">
            <span className="material-symbols-outlined text-error text-3xl">link_off</span>
          </div>
          <h2 className="font-headline text-2xl font-extrabold text-on-surface mb-3">
            Invalid Invite Link
          </h2>
          <p className="text-on-surface-variant text-sm leading-relaxed">
            This invite link is invalid or has expired (links expire after 48 hours). Please ask your Admin to send a new invitation.
          </p>
          <a
            href="/"
            className="mt-8 inline-flex items-center gap-2 text-sm text-primary font-semibold hover:underline"
          >
            <span className="material-symbols-outlined text-base">arrow_back</span>
            Back to Login
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex">
      {/* ─── Left Brand Panel (same as Login) ─────────────── */}
      <div
        className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12 relative overflow-hidden brand-panel-gradient"
      >
        <div className="absolute -top-32 -left-32 w-96 h-96 rounded-full bg-white/5 blur-3xl" />
        <div className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full bg-[#38bdf8]/10 blur-3xl" />

        {/* Logo */}
        <div className="relative z-10 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center">
            <span className="material-symbols-outlined text-white text-xl">monitor_heart</span>
          </div>
          <span className="text-white font-headline text-2xl font-black tracking-tighter">
            MetronAI
          </span>
        </div>

        {/* Hero Text */}
        <div className="relative z-10 space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/10 border border-white/20">
            <span className="w-2 h-2 rounded-full bg-[#6bff8f] pulse-orb" />
            <span className="text-white/80 text-xs font-bold uppercase tracking-widest">
              You&apos;ve been invited
            </span>
          </div>
          <h1 className="font-headline text-4xl xl:text-5xl font-extrabold text-white leading-tight tracking-tight">
            Join your<br />
            team on<br />
            <span className="text-[#38bdf8]">MetronAI.</span>
          </h1>
          <p className="text-white/60 text-lg leading-relaxed max-w-sm">
            Your workspace is ready. Set up your account and start collaborating on AI quality assurance with your team.
          </p>
        </div>

        {/* Bottom trust badges */}
        <div className="relative z-10 flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-white/40 text-sm">verified_user</span>
            <span className="text-white/40 text-xs font-medium uppercase tracking-wider">
              SOC 2 Compliant
            </span>
          </div>
          <div className="w-px h-4 bg-white/20" />
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-white/40 text-sm">lock</span>
            <span className="text-white/40 text-xs font-medium uppercase tracking-wider">
              Enterprise SSO
            </span>
          </div>
        </div>
      </div>

      {/* ─── Right Register Form ───────────────────────────── */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12 bg-background animate-fade-in">
        {/* Mobile logo */}
        <div className="lg:hidden flex items-center gap-2 mb-10">
          <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
            <span className="material-symbols-outlined text-white text-lg">monitor_heart</span>
          </div>
          <span className="font-headline text-xl font-black text-primary tracking-tighter">
            MetronAI
          </span>
        </div>

        <div className="w-full max-w-md">
          {/* Invitation banner */}
          <div className="mb-8 p-4 rounded-2xl bg-surface-container-lowest border border-outline-variant/20 card-shadow">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="material-symbols-outlined text-primary text-lg">person_add</span>
              </div>
              <div className="flex-1">
                <p className="text-xs text-on-surface-variant font-medium mb-1">
                  Invited by <span className="font-bold text-on-surface">{invitedBy}</span>
                </p>
                <p className="text-sm font-semibold text-on-surface break-all">{emailParam}</p>
                <div className={`mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${roleInfo.color}`}>
                  <span className="material-symbols-outlined text-sm">{roleInfo.icon}</span>
                  {roleInfo.label}
                </div>
              </div>
            </div>
          </div>

          {/* Header */}
          <div className="mb-8">
            <h2 className="font-headline text-3xl font-extrabold text-on-surface tracking-tight mb-2">
              Complete your account
            </h2>
            <p className="text-on-surface-variant text-sm">
              Your email is pre-verified. Just set your name and a strong password.
            </p>
          </div>

          {/* Error */}
          {error && (
            <div className="mb-6 flex items-center gap-3 px-4 py-3 rounded-xl bg-error-container text-error text-sm font-medium">
              <span className="material-symbols-outlined text-base">error</span>
              {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleRegister} className="space-y-5">
            {/* Email (read-only) */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">
                Work Email
              </label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">
                  mail
                </span>
                <input
                  type="email"
                  aria-label="Work Email"
                  value={emailParam}
                  readOnly
                  className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-surface-container ring-1 ring-outline-variant/20 text-on-surface-variant text-sm cursor-not-allowed"
                />
                <span className="material-symbols-outlined absolute right-4 top-1/2 -translate-y-1/2 text-secondary text-lg">
                  verified
                </span>
              </div>
            </div>

            {/* Full Name */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">
                Full Name
              </label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">
                  badge
                </span>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Sarah Jenkins"
                  className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-primary-container focus:outline-none text-on-surface placeholder:text-outline/50 text-sm transition-all"
                  required
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">
                Create Password
              </label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">
                  lock
                </span>
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Min. 8 characters"
                  className="w-full pl-12 pr-12 py-3.5 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-primary-container focus:outline-none text-on-surface placeholder:text-outline/50 text-sm transition-all"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-outline hover:text-primary transition-colors"
                >
                  <span className="material-symbols-outlined text-xl">
                    {showPassword ? "visibility_off" : "visibility"}
                  </span>
                </button>
              </div>
            </div>

            {/* Confirm Password */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">
                  lock_reset
                </span>
                <input
                  type={showPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Repeat your password"
                  className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-primary-container focus:outline-none text-on-surface placeholder:text-outline/50 text-sm transition-all"
                  required
                />
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-4 rounded-xl btn-primary text-white font-headline font-bold text-sm shadow-lg shadow-primary/20 flex items-center justify-center gap-3 disabled:opacity-70"
            >
              {loading ? (
                <>
                  <span className="material-symbols-outlined animate-spin text-base">
                    progress_activity
                  </span>
                  Creating account...
                </>
              ) : (
                <>
                  Activate My Account
                  <span className="material-symbols-outlined text-base">bolt</span>
                </>
              )}
            </button>
          </form>

          <p className="mt-8 text-center text-xs text-on-surface-variant">
            Already have an account?{" "}
            <a href="/" className="font-semibold text-primary hover:underline underline-offset-4">
              Sign In
            </a>
          </p>

          <p className="mt-4 text-center text-[10px] text-outline/50 uppercase tracking-widest">
            MetronAI v1.0 · Enterprise Edition
          </p>
        </div>
      </div>
    </div>
  );
}

export default function RegisterPage() {
  return (
    <Suspense>
      <RegisterForm />
    </Suspense>
  );
}
