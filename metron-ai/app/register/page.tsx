"use client";

import { useState, Suspense } from "react";
import { signUp, confirmSignUp } from "aws-amplify/auth";

function RegisterForm() {
  const [step, setStep] = useState<"signup" | "confirm">("signup");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [code, setCode] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!name.trim()) { setError("Please enter your full name."); return; }
    if (password.length < 8) { setError("Password must be at least 8 characters."); return; }
    if (password !== confirmPassword) { setError("Passwords do not match."); return; }
    setLoading(true);
    try {
      await signUp({
        username: email.trim(),
        password,
        options: { userAttributes: { email: email.trim(), name: name.trim() } },
      });
      setStep("confirm");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Sign-up failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (code.length < 6) { setError("Please enter the 6-digit verification code."); return; }
    setLoading(true);
    try {
      await confirmSignUp({ username: email.trim(), confirmationCode: code.trim() });
      window.location.href = "/?registered=1";
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Invalid code. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* ─── Left Brand Panel ─────────────────────────────────────── */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12 relative overflow-hidden brand-panel-gradient">
        <div className="absolute -top-32 -left-32 w-96 h-96 rounded-full bg-white/5 blur-3xl" />
        <div className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full bg-[#38bdf8]/10 blur-3xl" />

        <div className="relative z-10 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center">
            <span className="material-symbols-outlined text-white text-xl">monitor_heart</span>
          </div>
          <span className="text-white font-headline text-2xl font-black tracking-tighter">MetronAI</span>
        </div>

        <div className="relative z-10 space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/10 border border-white/20">
            <span className="w-2 h-2 rounded-full bg-[#6bff8f] pulse-orb" />
            <span className="text-white/80 text-xs font-bold uppercase tracking-widest">
              {step === "signup" ? "Create your account" : "Verify your email"}
            </span>
          </div>
          <h1 className="font-headline text-4xl xl:text-5xl font-extrabold text-white leading-tight tracking-tight">
            {step === "signup" ? (
              <>Join your<br />team on<br /><span className="text-[#38bdf8]">MetronAI.</span></>
            ) : (
              <>Check your<br />inbox for<br /><span className="text-[#38bdf8]">the code.</span></>
            )}
          </h1>
          <p className="text-white/60 text-lg leading-relaxed max-w-sm">
            {step === "signup"
              ? "Set up your account and start collaborating on AI quality assurance."
              : `We sent a 6-digit code to ${email}. Enter it to activate your account.`}
          </p>
        </div>

        <div className="relative z-10 flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-white/40 text-sm">verified_user</span>
            <span className="text-white/40 text-xs font-medium uppercase tracking-wider">SOC 2 Compliant</span>
          </div>
          <div className="w-px h-4 bg-white/20" />
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-white/40 text-sm">lock</span>
            <span className="text-white/40 text-xs font-medium uppercase tracking-wider">Enterprise SSO</span>
          </div>
        </div>
      </div>

      {/* ─── Right Form ───────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12 bg-background animate-fade-in">
        <div className="lg:hidden flex items-center gap-2 mb-10">
          <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
            <span className="material-symbols-outlined text-white text-lg">monitor_heart</span>
          </div>
          <span className="font-headline text-xl font-black text-primary tracking-tighter">MetronAI</span>
        </div>

        <div className="w-full max-w-md">
          {/* Step indicator */}
          <div className="flex items-center gap-3 mb-8">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${step === "signup" ? "bg-primary text-white" : "bg-secondary/20 text-secondary"}`}>
              {step === "confirm" ? <span className="material-symbols-outlined text-sm">check</span> : "1"}
            </div>
            <div className="flex-1 h-px bg-outline-variant/30" />
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${step === "confirm" ? "bg-primary text-white" : "bg-surface-container text-outline"}`}>2</div>
          </div>

          <div className="mb-8">
            <h2 className="font-headline text-3xl font-extrabold text-on-surface tracking-tight mb-2">
              {step === "signup" ? "Create your account" : "Verify your email"}
            </h2>
            <p className="text-on-surface-variant text-sm">
              {step === "signup"
                ? "Sign up with your work email to get started."
                : `Enter the 6-digit code sent to ${email}.`}
            </p>
          </div>

          {error && (
            <div className="mb-6 flex items-center gap-3 px-4 py-3 rounded-xl bg-error-container text-error text-sm font-medium">
              <span className="material-symbols-outlined text-base">error</span>
              {error}
            </div>
          )}

          {step === "signup" ? (
            <form onSubmit={handleSignUp} className="space-y-5">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">Work Email</label>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">mail</span>
                  <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@company.com"
                    className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-primary-container focus:outline-none text-on-surface placeholder:text-outline/50 text-sm transition-all" required />
                </div>
              </div>
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">Full Name</label>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">badge</span>
                  <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Sarah Jenkins"
                    className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-primary-container focus:outline-none text-on-surface placeholder:text-outline/50 text-sm transition-all" required />
                </div>
              </div>
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">Create Password</label>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">lock</span>
                  <input type={showPassword ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Min. 8 characters"
                    className="w-full pl-12 pr-12 py-3.5 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-primary-container focus:outline-none text-on-surface placeholder:text-outline/50 text-sm transition-all" required />
                  <button type="button" onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-outline hover:text-primary transition-colors">
                    <span className="material-symbols-outlined text-xl">{showPassword ? "visibility_off" : "visibility"}</span>
                  </button>
                </div>
              </div>
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">Confirm Password</label>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">lock_reset</span>
                  <input type={showPassword ? "text" : "password"} value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Repeat your password"
                    className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-primary-container focus:outline-none text-on-surface placeholder:text-outline/50 text-sm transition-all" required />
                </div>
              </div>
              <button type="submit" disabled={loading}
                className="w-full py-4 rounded-xl btn-primary text-white font-headline font-bold text-sm shadow-lg shadow-primary/20 flex items-center justify-center gap-3 disabled:opacity-70">
                {loading
                  ? <><span className="material-symbols-outlined animate-spin text-base">progress_activity</span>Creating account...</>
                  : <>Create Account<span className="material-symbols-outlined text-base">arrow_forward</span></>}
              </button>
            </form>
          ) : (
            <form onSubmit={handleConfirm} className="space-y-5">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-on-surface-variant mb-2">Verification Code</label>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-xl">pin</span>
                  <input type="text" inputMode="numeric" maxLength={6} value={code}
                    onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
                    placeholder="123456"
                    className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-surface-container-low ring-1 ring-outline-variant/30 focus:ring-2 focus:ring-primary-container focus:outline-none text-on-surface placeholder:text-outline/50 text-sm tracking-[0.3em] transition-all" required />
                </div>
              </div>
              <button type="submit" disabled={loading}
                className="w-full py-4 rounded-xl btn-primary text-white font-headline font-bold text-sm shadow-lg shadow-primary/20 flex items-center justify-center gap-3 disabled:opacity-70">
                {loading
                  ? <><span className="material-symbols-outlined animate-spin text-base">progress_activity</span>Verifying...</>
                  : <>Activate My Account<span className="material-symbols-outlined text-base">bolt</span></>}
              </button>
              <button type="button" onClick={() => setStep("signup")}
                className="w-full text-sm text-on-surface-variant hover:text-primary transition-colors">
                ← Back to sign up
              </button>
            </form>
          )}

          <p className="mt-8 text-center text-xs text-on-surface-variant">
            Already have an account?{" "}
            <a href="/" className="font-semibold text-primary hover:underline underline-offset-4">Sign In</a>
          </p>
          <p className="mt-4 text-center text-[10px] text-outline/50 uppercase tracking-widest">MetronAI v1.0 · Enterprise Edition</p>
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
