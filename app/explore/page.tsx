"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import ChartsPanel from "@/components/ChartsPanel";
import DataTable, { Sort } from "@/components/DataTable";
import FilterPanel, { Meta } from "@/components/FilterPanel";
import MultiSelect from "@/components/MultiSelect";
import SummaryPanel from "@/components/SummaryPanel";
import { fmtInt, label as colLabel } from "@/lib/format";
import {
  EMPTY_FILTERS,
  ExplorerFilters,
  activeFilterCount,
  filterParams,
} from "@/lib/state";

const DATASET_TABS = [
  { id: "main", name: "Pathology reports" },
  { id: "cytology", name: "Cytology" },
] as const;

const VIEW_TABS = [
  { id: "explore", name: "Search" },
  { id: "charts", name: "Charts" },
  { id: "summary", name: "Data summary" },
] as const;

const PUBLIC_COLUMNS = [
  "category", "breed", "sex", "age", "age_text", "diagnosis", "tissues",
  "stains", "stains_charge", "charge_type", "diagnosis_category",
  "specific_lesions",
];

const PUBLIC_DEFAULT_VISIBLE = [
  "category", "breed", "sex", "age", "diagnosis", "tissues",
  "diagnosis_category", "specific_lesions",
];

const EXPORT_LIMIT = 1000;

const SEARCHABLE = [
  "category", "breed", "sex", "age_text", "diagnosis", "stains",
  "charge_type", "diagnosis_category", "specific_lesions",
];

export default function ExplorePage() {
  const [filters, setFilters] = useState<ExplorerFilters>(EMPTY_FILTERS);
  const [view, setView] = useState<(typeof VIEW_TABS)[number]["id"]>("explore");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [sort, setSort] = useState<Sort | null>(null);
  const [customCols, setCustomCols] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [accessCode, setAccessCode] = useState("");
  const [showCodeInput, setShowCodeInput] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [dlError, setDlError] = useState<string | null>(null);
  const pageSize = 50;
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("drury_access_code");
      if (saved) setAccessCode(saved);
    } catch {
      /* storage unavailable */
    }
  }, []);

  const saveAccessCode = (v: string) => {
    setAccessCode(v);
    try {
      if (v) localStorage.setItem("drury_access_code", v);
      else localStorage.removeItem("drury_access_code");
    } catch {
      /* storage unavailable */
    }
  };

  const admin = meta?.admin ?? false;
  const allColumns = useMemo(
    () => (admin ? ["animal_name", ...PUBLIC_COLUMNS] : PUBLIC_COLUMNS),
    [admin]
  );
  const searchable = useMemo(
    () => (admin ? ["animal_name", ...SEARCHABLE] : SEARCHABLE),
    [admin]
  );
  const visibleCols = useMemo(() => {
    const base =
      customCols ?? (admin ? ["animal_name", ...PUBLIC_DEFAULT_VISIBLE] : PUBLIC_DEFAULT_VISIBLE);
    return base.filter((c) => allColumns.includes(c));
  }, [customCols, admin, allColumns]);

  // Dataset metadata (facet values, counts, admin flag).
  useEffect(() => {
    setMeta(null);
    setMetaError(null);
    fetch(`/api/meta?dataset=${filters.dataset}`)
      .then(async (r) => {
        const j = await r.json();
        if (!r.ok) throw new Error(j.error ?? "Failed to load dataset metadata");
        setMeta(j);
      })
      .catch((e) => setMetaError(e.message));
  }, [filters.dataset]);

  // Records (debounced; superseded requests aborted).
  useEffect(() => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    const p = filterParams(filters);
    p.set("page", String(page));
    p.set("pageSize", String(pageSize));
    if (sort) {
      p.set("sort", sort.col);
      p.set("dir", sort.dir);
    }
    const t = setTimeout(() => {
      fetch(`/api/records?${p.toString()}`, { signal: ctrl.signal })
        .then(async (r) => {
          const j = await r.json();
          if (!r.ok) throw new Error(j.error ?? "Request failed");
          setRows(j.rows);
          setTotal(j.total);
          setLoading(false);
        })
        .catch((e) => {
          if (e.name !== "AbortError") setLoading(false);
        });
    }, 300);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [filters, page, sort]);

  const onFilters = useCallback((next: ExplorerFilters) => {
    setFilters(next);
    setPage(1);
  }, []);

  const switchDataset = (ds: "main" | "cytology") => {
    setFilters({ ...EMPTY_FILTERS, dataset: ds });
    setPage(1);
    setSort(null);
  };

  const onSort = (col: string) => {
    setSort((s) =>
      s?.col === col
        ? s.dir === "asc"
          ? { col, dir: "desc" }
          : null
        : { col, dir: "asc" }
    );
    setPage(1);
  };

  const exportUrl = useMemo(() => {
    const p = filterParams(filters);
    if (accessCode.trim()) p.set("code", accessCode.trim());
    return `/api/export?${p.toString()}`;
  }, [filters, accessCode]);

  const exportEligible =
    admin ||
    accessCode.trim().length > 0 ||
    (activeFilterCount(filters) > 0 && total <= EXPORT_LIMIT);

  const doDownload = async () => {
    setDownloading(true);
    setDlError(null);
    try {
      const r = await fetch(exportUrl);
      if (!r.ok) {
        const j = await r.json().catch(() => null);
        throw new Error(j?.error ?? "Download failed");
      }
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `drury_${filters.dataset}_export.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      setDlError(e instanceof Error ? e.message : "Download failed");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1500px] px-4 py-6 lg:px-8">
      <header className="mb-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight">
            <Link href="/">Drury R. Reavill Pathology Database</Link>
          </h1>
          <p className="text-sm" style={{ color: "var(--ink-2)" }}>
            Exotic companion animal pathology archive · UC Davis
            {admin && (
              <span className="chip ml-2" title="Patient names are visible">
                Admin
              </span>
            )}
          </p>
        </div>
        <div
          className="flex items-center gap-1 rounded-lg p-1"
          style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
          {DATASET_TABS.map((d) => (
            <button
              key={d.id}
              type="button"
              className="rounded-md px-3 py-1.5 text-sm font-medium"
              style={
                filters.dataset === d.id
                  ? { background: "var(--accent)", color: "var(--accent-ink)" }
                  : { color: "var(--ink-2)" }
              }
              onClick={() => switchDataset(d.id)}
            >
              {d.name}
            </button>
          ))}
        </div>
      </header>

      {metaError ? (
        <div className="card mx-auto mt-16 max-w-lg p-8 text-center">
          <h2 className="mb-2 text-lg font-semibold">Database not available</h2>
          <p className="text-sm" style={{ color: "var(--ink-2)" }}>
            {metaError}
          </p>
          <p className="mt-3 text-sm" style={{ color: "var(--muted)" }}>
            Set <code>DATABASE_URL</code> to a Neon Postgres connection string
            and run <code>npm run db:seed</code>. See the README for setup steps.
          </p>
        </div>
      ) : (
        <>
          <nav
            className="mb-4 flex gap-1 border-b"
            style={{ borderColor: "var(--grid)" }}
            aria-label="Views"
          >
            {VIEW_TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                className="tab"
                data-active={view === t.id}
                onClick={() => setView(t.id)}
              >
                {t.name}
              </button>
            ))}
          </nav>

          {view === "summary" ? (
            <SummaryPanel dataset={filters.dataset} />
          ) : (
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-[300px_1fr]">
              <aside>
                <FilterPanel
                  filters={filters}
                  meta={meta}
                  searchable={searchable}
                  onChange={onFilters}
                />
              </aside>
              <main className="min-w-0">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <p className="text-sm" style={{ color: "var(--ink-2)" }}>
                    <b className="tabular" style={{ color: "var(--ink)" }}>
                      {fmtInt(total)}
                    </b>{" "}
                    of{" "}
                    <span className="tabular">{meta ? fmtInt(meta.total) : "…"}</span>{" "}
                    records match
                  </p>
                  {view === "explore" && (
                    <div className="flex items-center gap-2">
                      <div className="w-56">
                        <MultiSelect
                          label=""
                          options={allColumns.map((c) => ({ value: c, name: colLabel(c) }))}
                          selected={visibleCols}
                          onChange={(v) => setCustomCols(v.length ? v : null)}
                          placeholder="Columns to display"
                        />
                      </div>
                      {exportEligible ? (
                        <button
                          type="button"
                          className="btn btn-primary"
                          onClick={doDownload}
                          disabled={downloading}
                        >
                          {downloading ? "Preparing…" : "Download CSV"}
                        </button>
                      ) : (
                        <span
                          className="btn"
                          style={{ opacity: 0.55, cursor: "not-allowed" }}
                          title={
                            activeFilterCount(filters) === 0
                              ? "Apply a search or filter first"
                              : `Downloads are limited to ${EXPORT_LIMIT.toLocaleString()} records`
                          }
                        >
                          Download CSV
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {view === "explore" && (dlError || !admin) && (
                  <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
                    {dlError && (
                      <p className="w-full" style={{ color: "var(--series-8)" }}>
                        {dlError}
                      </p>
                    )}
                    {!admin &&
                      (showCodeInput || accessCode ? (
                        <span className="flex items-center gap-2">
                          <span className="font-medium" style={{ color: "var(--ink-2)" }}>
                            Download access code
                          </span>
                          <input
                            className="input"
                            style={{ width: "13rem", padding: "0.25rem 0.5rem" }}
                            value={accessCode}
                            onChange={(e) =>
                              saveAccessCode(e.target.value.toUpperCase().trim())
                            }
                            placeholder="e.g. 3F62A9C48B17D05E"
                          />
                          {accessCode && (
                            <button
                              type="button"
                              style={{ color: "var(--accent)" }}
                              onClick={() => {
                                saveAccessCode("");
                                setShowCodeInput(false);
                              }}
                            >
                              Clear
                            </button>
                          )}
                        </span>
                      ) : (
                        <button
                          type="button"
                          style={{ color: "var(--accent)" }}
                          onClick={() => setShowCodeInput(true)}
                        >
                          Have a download access code?
                        </button>
                      ))}
                  </div>
                )}
                {view === "explore" ? (
                  <DataTable
                    rows={rows}
                    columns={visibleCols}
                    sort={sort}
                    onSort={onSort}
                    page={page}
                    pageSize={pageSize}
                    total={total}
                    onPage={setPage}
                    loading={loading}
                  />
                ) : (
                  <ChartsPanel filters={filters} />
                )}
              </main>
            </div>
          )}
        </>
      )}

      <footer
        className="mt-10 border-t pt-4 text-xs"
        style={{ borderColor: "var(--grid)", color: "var(--muted)" }}
      >
        Drury R. Reavill Pathology Database at UC Davis · patient identities are
        anonymized for public access · CSV downloads are limited to search
        results under {EXPORT_LIMIT.toLocaleString()} records — for larger
        extracts email{" "}
        <a href="mailto:hbeaufrere@ucdavis.edu" style={{ color: "var(--accent)" }}>
          hbeaufrere@ucdavis.edu
        </a>{" "}
        with the purpose of your request ·{" "}
        <Link href="/#cite" style={{ color: "var(--accent)" }}>
          How to cite
        </Link>
      </footer>
    </div>
  );
}
