"use client";

import { fmtInt } from "@/lib/format";
import { label } from "@/lib/format";

export interface Sort {
  col: string;
  dir: "asc" | "desc";
}

const PAGE_SIZES = [25, 50, 100, 200];

export default function DataTable({
  rows,
  columns,
  sort,
  onSort,
  page,
  pageSize,
  total,
  onPage,
  onPageSize,
  loading,
}: {
  rows: Record<string, unknown>[];
  columns: string[];
  sort: Sort | null;
  onSort: (col: string) => void;
  page: number;
  pageSize: number;
  total: number;
  onPage: (p: number) => void;
  onPageSize: (n: number) => void;
  loading: boolean;
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(total, page * pageSize);

  return (
    <div className="flex flex-col gap-3">
      <div
        className="table-wrap"
        style={{
          // Fill the rest of the viewport (header + toolbar + pagination
          // take ~19rem) but never collapse below a usable height.
          maxHeight: "max(24rem, calc(100vh - 19rem))",
          opacity: loading ? 0.6 : 1,
        }}
      >
        <table className="data">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c} onClick={() => onSort(c)} title="Click to sort">
                  <span className="inline-flex items-center gap-1">
                    {label(c)}
                    {sort?.col === c && (
                      <span aria-hidden style={{ color: "var(--accent)" }}>
                        {sort.dir === "asc" ? "▲" : "▼"}
                      </span>
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={(r.id as number) ?? i}>
                {columns.map((c) => (
                  <td
                    key={c}
                    className={c === "age" || c === "tissues" || c === "stains_charge" ? "tabular" : ""}
                  >
                    {r[c] === null || r[c] === undefined || r[c] === "" ? (
                      <span style={{ color: "var(--muted)" }}>—</span>
                    ) : (
                      String(r[c])
                    )}
                  </td>
                ))}
              </tr>
            ))}
            {rows.length === 0 && !loading && (
              <tr>
                <td colSpan={columns.length} className="py-8 text-center" style={{ color: "var(--muted)" }}>
                  No records match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <span className="flex flex-wrap items-center gap-4" style={{ color: "var(--ink-2)" }}>
          <span>
            Showing <b className="tabular">{fmtInt(from)}–{fmtInt(to)}</b> of{" "}
            <b className="tabular">{fmtInt(total)}</b> rows
          </span>
          <label className="flex items-center gap-1.5">
            Rows per page
            <select
              className="input"
              style={{ width: "auto", padding: "0.25rem 0.5rem" }}
              value={pageSize}
              onChange={(e) => onPageSize(Number(e.target.value))}
            >
              {PAGE_SIZES.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </span>
        <span className="inline-flex items-center gap-1">
          <button className="btn" disabled={page <= 1} onClick={() => onPage(1)}>
            «
          </button>
          <button className="btn" disabled={page <= 1} onClick={() => onPage(page - 1)}>
            Prev
          </button>
          <span className="tabular px-2" style={{ color: "var(--ink-2)" }}>
            {fmtInt(page)} / {fmtInt(pages)}
          </span>
          <button className="btn" disabled={page >= pages} onClick={() => onPage(page + 1)}>
            Next
          </button>
          <button className="btn" disabled={page >= pages} onClick={() => onPage(pages)}>
            »
          </button>
        </span>
      </div>
    </div>
  );
}
