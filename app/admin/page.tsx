"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";

interface DownloadCode {
  code: string;
  note: string | null;
  created_at: string;
  expires_at: string | null;
  revoked: boolean;
}

export default function AdminPage() {
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"unknown" | "in" | "out">("unknown");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [codes, setCodes] = useState<DownloadCode[] | null>(null);
  const [codesError, setCodesError] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [expiresDays, setExpiresDays] = useState("");
  const [creating, setCreating] = useState(false);
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const loadCodes = useCallback(async () => {
    try {
      const r = await fetch("/api/admin/codes");
      const j = await r.json();
      if (!r.ok) throw new Error(j.error ?? "Failed to load codes");
      setCodes(j.codes);
      setCodesError(null);
    } catch (e) {
      setCodesError(e instanceof Error ? e.message : "Failed to load codes");
    }
  }, []);

  useEffect(() => {
    fetch("/api/meta?dataset=main")
      .then((r) => r.json())
      .then((j) => {
        setStatus(j.admin ? "in" : "out");
        if (j.admin) loadCodes();
      })
      .catch(() => setStatus("out"));
  }, [loadCodes]);

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
      loadCodes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  };

  const logout = async () => {
    await fetch("/api/admin/login", { method: "DELETE" });
    setStatus("out");
    setCodes(null);
  };

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    try {
      const d = Number(expiresDays);
      const r = await fetch("/api/admin/codes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          note,
          expiresDays: Number.isFinite(d) && d > 0 ? d : null,
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error ?? "Failed to create code");
      setNote("");
      setExpiresDays("");
      await loadCodes();
    } catch (e2) {
      setCodesError(e2 instanceof Error ? e2.message : "Failed to create code");
    } finally {
      setCreating(false);
    }
  };

  const revoke = async (code: string) => {
    await fetch(`/api/admin/codes?code=${encodeURIComponent(code)}`, { method: "DELETE" });
    await loadCodes();
  };

  const copy = async (code: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCode(code);
      setTimeout(() => setCopiedCode(null), 1500);
    } catch {
      /* selectable text remains */
    }
  };

  const fmtDate = (s: string | null) =>
    s ? new Date(s).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" }) : null;

  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <div className="card p-8">
        <h1 className="mb-1 text-lg font-bold">Admin access</h1>
        <p className="mb-6 text-sm" style={{ color: "var(--ink-2)" }}>
          Signing in reveals patient names in the explorer and CSV exports, and
          lets you manage download access codes. Public visitors always see
          anonymized records.
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
          <form onSubmit={login} className="flex max-w-sm flex-col gap-3">
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
      </div>

      {status === "in" && (
        <div className="card mt-6 p-8">
          <h2 className="mb-1 text-base font-bold">Download access codes</h2>
          <p className="mb-5 text-sm" style={{ color: "var(--ink-2)" }}>
            When someone emails you requesting a large extract, create a code
            here and send it to them. Entered in the explorer (&ldquo;Have a
            download access code?&rdquo;), it unlocks CSV downloads above the
            1,000-record limit — data stays anonymized. Revoke a code any time.
          </p>

          <form onSubmit={create} className="mb-6 flex flex-wrap items-end gap-3">
            <label className="flex flex-1 min-w-52 flex-col gap-1 text-xs font-medium" style={{ color: "var(--ink-2)" }}>
              Note (who / purpose)
              <input
                className="input"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="e.g. Dr. Smith — ferret adrenal study"
              />
            </label>
            <label className="flex w-32 flex-col gap-1 text-xs font-medium" style={{ color: "var(--ink-2)" }}>
              Expires (days)
              <input
                className="input"
                type="number"
                min={1}
                value={expiresDays}
                onChange={(e) => setExpiresDays(e.target.value)}
                placeholder="never"
              />
            </label>
            <button className="btn btn-primary" disabled={creating}>
              {creating ? "Creating…" : "Create code"}
            </button>
          </form>

          {codesError && (
            <p className="mb-3 text-sm" style={{ color: "var(--series-8)" }}>
              {codesError}
            </p>
          )}

          {codes === null ? (
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              Loading…
            </p>
          ) : codes.length === 0 ? (
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              No codes issued yet.
            </p>
          ) : (
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th style={{ cursor: "default" }}>Code</th>
                    <th style={{ cursor: "default" }}>Note</th>
                    <th style={{ cursor: "default" }}>Created</th>
                    <th style={{ cursor: "default" }}>Expires</th>
                    <th style={{ cursor: "default" }}>Status</th>
                    <th style={{ cursor: "default" }} />
                  </tr>
                </thead>
                <tbody>
                  {codes.map((c) => {
                    const expired =
                      c.expires_at !== null && new Date(c.expires_at) < new Date();
                    return (
                      <tr key={c.code}>
                        <td className="tabular font-medium">{c.code}</td>
                        <td>{c.note ?? <span style={{ color: "var(--muted)" }}>—</span>}</td>
                        <td className="tabular">{fmtDate(c.created_at)}</td>
                        <td className="tabular">
                          {fmtDate(c.expires_at) ?? (
                            <span style={{ color: "var(--muted)" }}>never</span>
                          )}
                        </td>
                        <td>
                          {c.revoked ? (
                            <span style={{ color: "var(--series-8)" }}>revoked</span>
                          ) : expired ? (
                            <span style={{ color: "var(--muted)" }}>expired</span>
                          ) : (
                            <span style={{ color: "var(--series-6)" }}>active</span>
                          )}
                        </td>
                        <td>
                          <span className="flex gap-2">
                            <button
                              type="button"
                              className="text-xs font-medium"
                              style={{ color: "var(--accent)" }}
                              onClick={() => copy(c.code)}
                            >
                              {copiedCode === c.code ? "Copied ✓" : "Copy"}
                            </button>
                            {!c.revoked && !expired && (
                              <button
                                type="button"
                                className="text-xs font-medium"
                                style={{ color: "var(--series-8)" }}
                                onClick={() => revoke(c.code)}
                              >
                                Revoke
                              </button>
                            )}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <p className="mt-6 text-xs" style={{ color: "var(--muted)" }}>
        <Link href="/" style={{ color: "var(--accent)" }}>
          ← Back to overview
        </Link>
      </p>
    </div>
  );
}
