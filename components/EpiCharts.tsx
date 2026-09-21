// Server-rendered SVG/HTML charts for the epidemiology report.
// Palette slots are the validated taxonomic-class set (globals.css --cls-*).

import { fmtInt } from "@/lib/format";

export const CLS_COLOR: Record<string, string> = {
  Birds: "var(--cls-1)",
  Mammals: "var(--cls-2)",
  Reptiles: "var(--cls-3)",
  Fish: "var(--cls-4)",
  Amphibians: "var(--cls-5)",
  Invertebrates: "var(--cls-6)",
};

const SLOT = [
  "var(--cls-1)", "var(--cls-2)", "var(--cls-3)",
  "var(--cls-4)", "var(--cls-5)", "var(--cls-6)",
];

export function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      style={{
        width: 10, height: 10, borderRadius: 3, background: color,
        display: "inline-block", flexShrink: 0,
      }}
    />
  );
}

/* ---- 100% stacked horizontal bars (one row per class) ---- */
export function StackedShare({
  rows,
  keys,
  extraKey = "Other",
}: {
  rows: Record<string, unknown>[];
  keys: string[];
  extraKey?: string | null;
}) {
  const allKeys = extraKey ? [...keys, extraKey] : keys;
  const colorFor = (i: number) =>
    allKeys[i] === "Other" || (extraKey && i === allKeys.length - 1)
      ? "var(--muted)"
      : SLOT[i % SLOT.length];
  return (
    <div className="flex flex-col gap-3">
      {rows.map((r) => {
        const known = keys.reduce((s, k) => s + (Number(r[k]) || 0), 0);
        const rest = extraKey ? Math.max(0, 100 - known) : 0;
        const segs = [
          ...keys.map((k, i) => ({ k, v: Number(r[k]) || 0, c: colorFor(i) })),
          ...(extraKey ? [{ k: extraKey, v: rest, c: "var(--muted)" }] : []),
        ];
        return (
          <div key={String(r.cls)}>
            <div className="mb-1 flex items-baseline justify-between text-sm">
              <span className="font-medium">{String(r.cls)}</span>
              <span className="tabular text-xs" style={{ color: "var(--muted)" }}>
                n={fmtInt(Number(r.n))}
              </span>
            </div>
            <div
              className="flex overflow-hidden"
              style={{ height: 18, borderRadius: 5, gap: 2 }}
              role="img"
              aria-label={segs.map((s) => `${s.k} ${s.v}%`).join(", ")}
            >
              {segs.map(
                (s) =>
                  s.v > 0 && (
                    <div
                      key={s.k}
                      title={`${s.k}: ${s.v}%`}
                      style={{
                        width: `${s.v}%`, background: s.c, borderRadius: 3,
                        minWidth: s.v > 0 ? 2 : 0,
                      }}
                    />
                  )
              )}
            </div>
          </div>
        );
      })}
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs" style={{ color: "var(--ink-2)" }}>
        {allKeys.map((k, i) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <Swatch color={colorFor(i)} />
            {k}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ---- Multi-series line chart: tumor share vs age ---- */
export function TumorAgeLines({
  rows,
}: {
  rows: Record<string, number | string | null>[];
}) {
  const seriesDefs = [
    { key: "All classes", color: "var(--ink)", w: 2.5 },
    { key: "Birds", color: "var(--cls-1)", w: 2 },
    { key: "Mammals", color: "var(--cls-2)", w: 2 },
    { key: "Reptiles", color: "var(--cls-3)", w: 2 },
  ];
  const W = 640, H = 300, L = 44, R = 110, T = 14, B = 40;
  const maxY = 70;
  const x = (i: number) => L + (i * (W - L - R)) / (rows.length - 1);
  const y = (v: number) => T + (1 - v / maxY) * (H - T - B);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
         aria-label="Tumor share of diagnoses by age group" style={{ maxWidth: 720 }}>
      {[0, 20, 40, 60].map((g) => (
        <g key={g}>
          <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="var(--grid)" />
          <text x={L - 6} y={y(g) + 4} textAnchor="end" fontSize="11" fill="var(--muted)">
            {g}%
          </text>
        </g>
      ))}
      {rows.map((r, i) => (
        <text key={i} x={x(i)} y={H - B + 18} textAnchor="middle" fontSize="11" fill="var(--muted)">
          {String(r.bin)}
        </text>
      ))}
      <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill="var(--ink-2)">
        Age group (years)
      </text>
      {(() => {
        const series = seriesDefs.map((s) => {
          const pts = rows
            .map((r, i) => ({ i, v: r[s.key] as number | null }))
            .filter((p) => p.v !== null && p.v !== undefined);
          const last = pts[pts.length - 1];
          return { ...s, pts, last, labelY: last ? y(last.v as number) : 0 };
        });
        // Nudge end-labels apart so converging lines stay readable.
        const sorted = [...series].filter((s) => s.last).sort((a, b) => a.labelY - b.labelY);
        for (let i = 1; i < sorted.length; i++) {
          if (sorted[i].labelY - sorted[i - 1].labelY < 15) {
            sorted[i].labelY = sorted[i - 1].labelY + 15;
          }
        }
        return series.map((s) => (
          <g key={s.key}>
            <path
              d={s.pts.map((p, j) => `${j === 0 ? "M" : "L"} ${x(p.i)} ${y(p.v as number)}`).join(" ")}
              fill="none" stroke={s.color} strokeWidth={s.w} strokeLinejoin="round"
            />
            {s.pts.map((p) => (
              <circle key={p.i} cx={x(p.i)} cy={y(p.v as number)} r={3} fill={s.color}
                      stroke="var(--surface)" strokeWidth="1.5">
                <title>{`${s.key}, ${rows[p.i].bin} yrs: ${p.v}%`}</title>
              </circle>
            ))}
            {s.last && (
              <text x={x(s.last.i) + 8} y={s.labelY + 4} fontSize="11.5"
                    fontWeight="600" fill={s.color}>
                {s.key}
              </text>
            )}
          </g>
        ));
      })()}
    </svg>
  );
}

/* ---- Horizontal bar list (shared) ---- */
export function HBarList({
  rows,
  suffix = "",
  colorOf,
}: {
  rows: { label: string; value: number; note?: string; color?: string }[];
  suffix?: string;
  colorOf?: (r: { label: string; color?: string }) => string;
}) {
  const max = Math.max(...rows.map((r) => r.value), 1);
  return (
    <ul className="flex flex-col gap-2">
      {rows.map((r) => (
        <li key={r.label} className="text-sm">
          <div className="mb-0.5 flex items-baseline justify-between gap-2">
            <span className="truncate">{r.label}</span>
            <span className="tabular shrink-0 text-xs" style={{ color: "var(--ink-2)" }}>
              {fmtInt(Math.round(r.value * 10) / 10 === Math.round(r.value) ? r.value : r.value)}
              {suffix}
              {r.note ? ` · ${r.note}` : ""}
            </span>
          </div>
          <div aria-hidden style={{ height: 7, borderRadius: 4, background: "var(--grid)", overflow: "hidden" }}>
            <div
              style={{
                width: `${Math.max(1.5, (r.value / max) * 100)}%`, height: "100%",
                borderRadius: 4,
                background: colorOf ? colorOf(r) : r.color ?? "var(--cls-1)",
              }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

/* ---- Association strength chart (log-scaled bars) ---- */
export function AssociationChart({
  rows,
}: {
  rows: { cls: string; species: string; condition: string; cases: number; pct: number; ratio: number }[];
}) {
  const maxLog = Math.log10(Math.max(...rows.map((r) => r.ratio)));
  return (
    <div className="flex flex-col gap-2.5">
      {rows.map((r) => (
        <div key={`${r.species}-${r.condition}`} className="text-sm">
          <div className="mb-0.5 flex items-baseline justify-between gap-2">
            <span className="inline-flex min-w-0 items-center gap-2">
              <Swatch color={CLS_COLOR[r.cls]} />
              <span className="truncate">
                <b>{r.species}</b> — {r.condition}
              </span>
            </span>
            <span className="tabular shrink-0 text-xs" style={{ color: "var(--ink-2)" }}>
              {r.cases} cases · {r.pct}% of species
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div aria-hidden className="flex-1"
                 style={{ height: 7, borderRadius: 4, background: "var(--grid)", overflow: "hidden" }}>
              <div
                style={{
                  width: `${Math.max(3, (Math.log10(r.ratio) / maxLog) * 100)}%`,
                  height: "100%", borderRadius: 4, background: CLS_COLOR[r.cls],
                }}
              />
            </div>
            <span className="tabular w-12 text-right text-xs font-semibold" style={{ color: "var(--ink)" }}>
              {r.ratio}×
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ---- Body-system percentage heatmap ---- */
export function SystemHeatmap({
  systems,
  rows,
}: {
  systems: string[];
  rows: Record<string, unknown>[];
}) {
  const max = Math.max(
    ...rows.flatMap((r) => systems.map((s) => Number(r[s]) || 0))
  );
  const SEQ = ["var(--seq-100)", "var(--seq-250)", "var(--seq-400)", "var(--seq-550)", "var(--seq-700)"];
  const color = (v: number) => {
    if (v <= 0) return "transparent";
    const t = v / max;
    return SEQ[Math.min(SEQ.length - 1, Math.floor(t * SEQ.length))];
  };
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "separate", borderSpacing: 2, fontSize: 12 }}>
        <thead>
          <tr>
            <th />
            {systems.map((s) => (
              <th key={s} style={{ color: "var(--ink-2)", fontWeight: 500, padding: "4px 6px", whiteSpace: "nowrap" }}>
                {s}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={String(r.cls)}>
              <th style={{ color: "var(--ink-2)", fontWeight: 500, textAlign: "right", padding: "2px 8px", whiteSpace: "nowrap" }}>
                {String(r.cls)}
              </th>
              {systems.map((s) => {
                const v = Number(r[s]) || 0;
                return (
                  <td key={s} className="tabular"
                      title={`${r.cls} × ${s}: ${v}% of classified diagnoses`}
                      style={{
                        background: color(v),
                        color: v / max >= 0.6 ? "#ffffff" : "var(--ink)",
                        textAlign: "center", minWidth: 52, padding: "5px 6px", borderRadius: 4,
                        border: v === 0 ? "1px solid var(--grid)" : "1px solid transparent",
                      }}>
                    {v > 0 ? `${v}%` : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
