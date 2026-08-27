"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

export default function AdminPage() {
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"unknown" | "in" | "out">("unknown");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/api/meta?dataset=main")
      .then((r) => r.json())
      .then((j) => setStatus(j.admin ? "in" : "out"))
      .catch(() => setStatus("out"));
  }, []);

  const login = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/admin/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error ?? "Login failed");
      setStatus("in");
      setPassword("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  };

  const logout = async () => {
    await fetch("/api/admin/login", { method: "DELETE" });
    setStatus("out");
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-4">
      <div className="card p-8">
        <h1 className="mb-1 text-lg font-bold">Admin access</h1>
        <p className="mb-6 text-sm" style={{ color: "var(--ink-2)" }}>
          Signing in reveals patient names in the explorer and CSV exports.
          Public visitors always see anonymized records.
        </p>

        {status === "in" ? (
          <div className="flex flex-col gap-4">
            <p className="text-sm font-medium" style={{ color: "var(--series-6)" }}>
              ✓ You are signed in as admin.
            </p>
            <div className="flex gap-2">
              <Link href="/explore" className="btn btn-primary">
                Open the explorer
              </Link>
              <button type="button" className="btn" onClick={logout}>
                Sign out
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={login} className="flex flex-col gap-3">
            <input
              className="input"
              type="password"
              placeholder="Admin password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoFocus
            />
            {error && (
              <p className="text-sm" style={{ color: "var(--series-8)" }}>
                {error}
              </p>
            )}
            <button className="btn btn-primary justify-center" disabled={busy || !password}>
              {busy ? "Signing in…" : "Sign in"}
            </button>
          </form>
        )}

        <p className="mt-6 text-xs" style={{ color: "var(--muted)" }}>
          <Link href="/" style={{ color: "var(--accent)" }}>
            ← Back to overview
          </Link>
        </p>
      </div>
    </div>
  );
}
