"use client";

import { useEffect, useState } from "react";
import { fmtInt, fmtNum, label } from "@/lib/format";

interface ColumnSummary {
  name: string;
  nonNull: number;
  distinct: number;
  numeric: { min: number | null; max: number | null; avg: number | null; median: number | null } | null;
}

export default function SummaryPanel({ dataset }: { dataset: string }) {
  const [summary, setSummary] = useState<{ total: number; columns: ColumnSummary[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSummary(null);
    fetch(`/api/summary?dataset=${dataset}`)
      .then(async (r) => {
        const j = await r.json();
        if (!r.ok) throw new Error(j.error ?? "Request failed");
        setSummary(j);
      })
      .catch((e) => setError(e.message));
  }, [dataset]);

  if (error)
    return (
      <p className="py-16 text-center text-sm" style={{ color: "var(--series-8)" }}>
        {error}
      </p>
    );
  if (!summary)
    return (
      <p className="py-16 text-center text-sm" style={{ color: "var(--muted)" }}>
        Loading…
      </p>
    );

  return (
    <div className="flex flex-col gap-4">
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th style={{ cursor: "default" }}>Column</th>
              <th style={{ cursor: "default" }}>Non-null</th>
              <th style={{ cursor: "default" }}>Complete</th>
              <th style={{ cursor: "default" }}>Distinct values</th>
              <th style={{ cursor: "default" }}>Min</th>
              <th style={{ cursor: "default" }}>Median</th>
              <th style={{ cursor: "default" }}>Mean</th>
              <th style={{ cursor: "default" }}>Max</th>
            </tr>
          </thead>
          <tbody>
            {summary.columns.map((c) => {
              const pct = summary.total ? (c.nonNull / summary.total) * 100 : 0;
              return (
                <tr key={c.name}>
                  <td className="font-medium">{label(c.name)}</td>
                  <td className="tabular">{fmtInt(c.nonNull)}</td>
                  <td>
                    <span className="inline-flex items-center gap-2">
                      <span
                        aria-hidden
                        style={{
                          width: 64,
                          height: 6,
                          borderRadius: 3,
                          background: "var(--grid)",
                          display: "inline-block",
                          position: "relative",
                          overflow: "hidden",
                        }}
                      >
                        <span
                          style={{
                            position: "absolute",
                            inset: 0,
                            width: `${pct}%`,
                            background: "var(--series-1)",
                            borderRadius: 3,
                          }}
                        />
                      </span>
                      <span className="tabular text-xs" style={{ color: "var(--ink-2)" }}>
                        {pct.toFixed(0)}%
                      </span>
                    </span>
                  </td>
                  <td className="tabular">{fmtInt(c.distinct)}</td>
                  <td className="tabular">{c.numeric?.min != null ? fmtNum(c.numeric.min) : "—"}</td>
                  <td className="tabular">{c.numeric?.median != null ? fmtNum(c.numeric.median) : "—"}</td>
                  <td className="tabular">{c.numeric?.avg != null ? fmtNum(c.numeric.avg) : "—"}</td>
                  <td className="tabular">{c.numeric?.max != null ? fmtNum(c.numeric.max) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs" style={{ color: "var(--muted)" }}>
        Ages above 200 years in the source spreadsheet were treated as data-entry
        errors and stored as unknown. Text fields were lightly normalized during
        import (whitespace, Excel line-break artifacts, obvious typos in sex and
        diagnosis category values); the original parquet files in the repository
        remain untouched.
      </p>
    </div>
  );
}
