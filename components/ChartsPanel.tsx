"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { NUMERIC_COLUMNS, TEXT_COLUMNS } from "@/lib/filters";
import { fmtInt, fmtNum, label } from "@/lib/format";
import { ExplorerFilters, filterParams } from "@/lib/state";

const SERIES = [
  "var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)",
  "var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)",
];

const SEQ = ["var(--seq-100)", "var(--seq-250)", "var(--seq-400)", "var(--seq-550)", "var(--seq-700)"];

type ChartKind = "bar" | "donut" | "histogram" | "box" | "scatter" | "heatmap";

const KINDS: { id: ChartKind; name: string }[] = [
  { id: "bar", name: "Bar" },
  { id: "donut", name: "Donut" },
  { id: "histogram", name: "Histogram" },
  { id: "box", name: "Box plot" },
  { id: "scatter", name: "Scatter" },
  { id: "heatmap", name: "Heatmap" },
];

const tooltipStyle = {
  background: "var(--surface)",
  border: "1px solid var(--baseline)",
  borderRadius: 8,
  color: "var(--ink)",
  fontSize: 13,
};

function Select({
  value,
  onChange,
  options,
  title,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; name: string }[];
  title: string;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium" style={{ color: "var(--ink-2)" }}>
      {title}
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.name}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumInput({
  value,
  onChange,
  title,
  min,
  max,
}: {
  value: number;
  onChange: (v: number) => void;
  title: string;
  min: number;
  max: number;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium" style={{ color: "var(--ink-2)" }}>
      {title}
      <input
        className="input"
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(Math.min(max, Math.max(min, n)));
        }}
      />
    </label>
  );
}

const textOptions = TEXT_COLUMNS.filter((c) => c !== "animal_name").map(
  (c) => ({ value: c, name: label(c) })
);
const numOptions = NUMERIC_COLUMNS.map((c) => ({ value: c, name: label(c) }));

export default function ChartsPanel({ filters }: { filters: ExplorerFilters }) {
  const [kind, setKind] = useState<ChartKind>("bar");
  const [xText, setXText] = useState("category");
  const [yText, setYText] = useState("diagnosis_category");
  const [xNum, setXNum] = useState("age");
  const [yNum, setYNum] = useState("tissues");
  const [topN, setTopN] = useState(15);
  const [bins, setBins] = useState(40);
  // Data is tagged with the chart kind it was fetched for, so a stale
  // payload from the previous chart type is never rendered by the new one.
  const [data, setData] = useState<{ kind: ChartKind; rows: unknown[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const url = useMemo(() => {
    const p = filterParams(filters);
    switch (kind) {
      case "bar":
      case "donut":
        p.set("type", "count");
        p.set("x", xText);
        p.set("topN", String(kind === "donut" ? Math.min(topN, 8) : topN));
        break;
      case "histogram":
        p.set("type", "histogram");
        p.set("x", xNum);
        p.set("bins", String(bins));
        break;
      case "box":
        p.set("type", "box");
        p.set("x", xText);
        p.set("y", xNum);
        p.set("topN", String(topN));
        break;
      case "scatter":
        p.set("type", "scatter");
        p.set("x", xNum);
        p.set("y", yNum);
        break;
      case "heatmap":
        p.set("type", "heatmap");
        p.set("x", xText);
        p.set("y", yText);
        p.set("topN", String(Math.min(topN, 25)));
        break;
    }
    return `/api/chart?${p.toString()}`;
  }, [filters, kind, xText, yText, xNum, yNum, topN, bins]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    const t = setTimeout(() => {
      fetch(url)
        .then(async (r) => {
          const j = await r.json();
          if (!r.ok) throw new Error(j.error ?? "Request failed");
          return j;
        })
        .then((j) => {
          if (alive) setData({ kind, rows: j.data ?? [] });
        })
        .catch((e) => {
          if (alive) setError(e.message);
        })
        .finally(() => {
          if (alive) setLoading(false);
        });
    }, 250);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [url, kind]);

  const rows = data && data.kind === kind ? data.rows : null;

  return (
    <div className="flex flex-col gap-4">
      <div className="card flex flex-wrap items-end gap-3 p-4">
        <Select
          title="Chart type"
          value={kind}
          onChange={(v) => setKind(v as ChartKind)}
          options={KINDS.map((k) => ({ value: k.id, name: k.name }))}
        />
        {(kind === "bar" || kind === "donut" || kind === "box" || kind === "heatmap") && (
          <Select
            title={kind === "heatmap" ? "Rows" : "Group by"}
            value={xText}
            onChange={setXText}
            options={textOptions}
          />
        )}
        {kind === "heatmap" && (
          <Select
            title="Columns"
            value={yText}
            onChange={setYText}
            options={textOptions.filter((o) => o.value !== xText)}
          />
        )}
        {(kind === "histogram" || kind === "box" || kind === "scatter") && (
          <Select
            title={kind === "box" ? "Value (numeric)" : "X (numeric)"}
            value={xNum}
            onChange={setXNum}
            options={numOptions}
          />
        )}
        {kind === "scatter" && (
          <Select
            title="Y (numeric)"
            value={yNum}
            onChange={setYNum}
            options={numOptions.filter((o) => o.value !== xNum)}
          />
        )}
        {kind !== "scatter" && kind !== "histogram" && (
          <NumInput
            title={kind === "donut" ? "Top N (max 8)" : "Top N"}
            value={kind === "donut" ? Math.min(topN, 8) : topN}
            onChange={setTopN}
            min={2}
            max={kind === "donut" ? 8 : kind === "heatmap" ? 25 : 50}
          />
        )}
        {kind === "histogram" && (
          <NumInput title="Bins" value={bins} onChange={setBins} min={5} max={200} />
        )}
      </div>

      <div className="card p-4" style={{ minHeight: 480 }}>
        {error ? (
          <p className="py-16 text-center text-sm" style={{ color: "var(--series-8)" }}>
            {error}
          </p>
        ) : rows === null ? (
          <p className="py-16 text-center text-sm" style={{ color: "var(--muted)" }}>
            Loading…
          </p>
        ) : rows.length === 0 ? (
          <p className="py-16 text-center text-sm" style={{ color: "var(--muted)" }}>
            No data for this combination.
          </p>
        ) : (
          <div style={{ opacity: loading ? 0.6 : 1 }}>
            {kind === "bar" && <CountBar data={rows as CountRow[]} col={xText} />}
            {kind === "donut" && <Donut data={rows as CountRow[]} col={xText} />}
            {kind === "histogram" && <Histogram data={rows as HistRow[]} col={xNum} />}
            {kind === "box" && <BoxPlot data={rows as BoxRow[]} x={xText} y={xNum} />}
            {kind === "scatter" && (
              <ScatterPlot data={rows as { x: number; y: number }[]} x={xNum} y={yNum} />
            )}
            {kind === "heatmap" && <Heatmap data={rows as HeatRow[]} x={xText} y={yText} />}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- Bar ---------- */

interface CountRow {
  value: string;
  count: number;
}

function CountBar({ data, col }: { data: CountRow[]; col: string }) {
  const h = Math.max(320, data.length * 28 + 60);
  return (
    <>
      <ChartTitle text={`Record count by ${label(col).toLowerCase()}`} />
      <ResponsiveContainer width="100%" height={h}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 48, top: 4 }}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis
            type="number"
            tick={{ fill: "var(--muted)", fontSize: 12 }}
            stroke="var(--baseline)"
            tickFormatter={(v: number) => fmtInt(v)}
          />
          <YAxis
            type="category"
            dataKey="value"
            width={190}
            tick={{ fill: "var(--ink-2)", fontSize: 12 }}
            stroke="var(--baseline)"
          />
          <Tooltip
            contentStyle={tooltipStyle}
            cursor={{ fill: "color-mix(in srgb, var(--accent) 8%, transparent)" }}
            formatter={(v) => [fmtInt(Number(v)), "Records"]}
          />
          <Bar
            dataKey="count"
            fill="var(--series-1)"
            radius={[0, 4, 4, 0]}
            barSize={16}
            label={{
              position: "right",
              fill: "var(--ink-2)",
              fontSize: 11,
              formatter: (v: unknown) => fmtInt(Number(v)),
            }}
          />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

/* ---------- Donut ---------- */

function Donut({ data, col }: { data: CountRow[]; col: string }) {
  const total = data.reduce((s, d) => s + d.count, 0);
  return (
    <>
      <ChartTitle text={`Share of records by ${label(col).toLowerCase()} (top ${data.length})`} />
      <div className="flex flex-wrap items-center justify-center gap-6">
        <PieChart width={380} height={380}>
          <Pie
            data={data}
            dataKey="count"
            nameKey="value"
            innerRadius={85}
            outerRadius={150}
            paddingAngle={1.5}
            stroke="var(--surface)"
            strokeWidth={2}
          >
            {data.map((d, i) => (
              <Cell key={d.value} fill={SERIES[i % SERIES.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v, name) => [
              `${fmtInt(Number(v))} (${((Number(v) / total) * 100).toFixed(1)}%)`,
              String(name),
            ]}
          />
        </PieChart>
        <ul className="flex flex-col gap-1.5 text-sm">
          {data.map((d, i) => (
            <li key={d.value} className="flex items-center gap-2">
              <span
                aria-hidden
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 3,
                  background: SERIES[i % SERIES.length],
                  display: "inline-block",
                }}
              />
              <span className="max-w-52 truncate">{d.value}</span>
              <span className="tabular" style={{ color: "var(--muted)" }}>
                {((d.count / total) * 100).toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

/* ---------- Histogram ---------- */

interface HistRow {
  x0: number;
  x1: number;
  count: number;
}

function Histogram({ data, col }: { data: HistRow[]; col: string }) {
  const rows = data.map((d) => ({
    ...d,
    mid: `${fmtNum(d.x0)}–${fmtNum(d.x1)}`,
  }));
  return (
    <>
      <ChartTitle text={`Distribution of ${label(col).toLowerCase()}`} />
      <ResponsiveContainer width="100%" height={420}>
        <BarChart data={rows} margin={{ left: 8, right: 16, top: 8 }} barCategoryGap={1}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis
            dataKey="mid"
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            stroke="var(--baseline)"
            minTickGap={30}
          />
          <YAxis
            tick={{ fill: "var(--muted)", fontSize: 12 }}
            stroke="var(--baseline)"
            tickFormatter={(v: number) => fmtInt(v)}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            cursor={{ fill: "color-mix(in srgb, var(--accent) 8%, transparent)" }}
            formatter={(v) => [fmtInt(Number(v)), "Records"]}
            labelFormatter={(l) => `${label(col)}: ${l}`}
          />
          <Bar dataKey="count" fill="var(--series-1)" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

/* ---------- Box plot (custom SVG) ---------- */

interface BoxRow {
  group: string;
  n: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
}

function BoxPlot({ data, x, y }: { data: BoxRow[]; x: string; y: string }) {
  const width = 900;
  const rowH = 34;
  const left = 200;
  const right = 60;
  const height = data.length * rowH + 40;
  const lo = Math.min(...data.map((d) => d.min));
  const hi = Math.max(...data.map((d) => d.max));
  const span = hi - lo || 1;
  const sx = (v: number) => left + ((v - lo) / span) * (width - left - right);

  const ticks = 6;
  const tickVals = Array.from({ length: ticks + 1 }, (_, i) => lo + (span * i) / ticks);

  return (
    <>
      <ChartTitle text={`${label(y)} by ${label(x).toLowerCase()} (box = quartiles, line = median)`} />
      <div style={{ overflowX: "auto" }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          style={{ minWidth: 640 }}
          role="img"
          aria-label={`Box plot of ${label(y)} by ${label(x)}`}
        >
          {tickVals.map((t, i) => (
            <g key={i}>
              <line x1={sx(t)} x2={sx(t)} y1={8} y2={height - 32} stroke="var(--grid)" />
              <text x={sx(t)} y={height - 14} textAnchor="middle" fontSize={11} fill="var(--muted)">
                {fmtNum(t)}
              </text>
            </g>
          ))}
          {data.map((d, i) => {
            const cy = 20 + i * rowH;
            return (
              <g key={d.group}>
                <title>
                  {`${d.group} — n=${fmtInt(d.n)}, min ${fmtNum(d.min)}, Q1 ${fmtNum(d.q1)}, median ${fmtNum(d.median)}, Q3 ${fmtNum(d.q3)}, max ${fmtNum(d.max)}`}
                </title>
                <text
                  x={left - 10}
                  y={cy + 4}
                  textAnchor="end"
                  fontSize={12}
                  fill="var(--ink-2)"
                >
                  {d.group.length > 26 ? d.group.slice(0, 25) + "…" : d.group}
                </text>
                <line x1={sx(d.min)} x2={sx(d.max)} y1={cy} y2={cy} stroke="var(--baseline)" strokeWidth={1.5} />
                <line x1={sx(d.min)} x2={sx(d.min)} y1={cy - 5} y2={cy + 5} stroke="var(--baseline)" strokeWidth={1.5} />
                <line x1={sx(d.max)} x2={sx(d.max)} y1={cy - 5} y2={cy + 5} stroke="var(--baseline)" strokeWidth={1.5} />
                <rect
                  x={sx(d.q1)}
                  y={cy - 8}
                  width={Math.max(2, sx(d.q3) - sx(d.q1))}
                  height={16}
                  rx={3}
                  fill="color-mix(in srgb, var(--series-1) 35%, var(--surface))"
                  stroke="var(--series-1)"
                  strokeWidth={1.5}
                />
                <line
                  x1={sx(d.median)}
                  x2={sx(d.median)}
                  y1={cy - 8}
                  y2={cy + 8}
                  stroke="var(--series-1)"
                  strokeWidth={2.5}
                />
              </g>
            );
          })}
        </svg>
      </div>
    </>
  );
}

/* ---------- Scatter ---------- */

function ScatterPlot({
  data,
  x,
  y,
}: {
  data: { x: number; y: number }[];
  x: string;
  y: string;
}) {
  return (
    <>
      <ChartTitle
        text={`${label(y)} vs ${label(x)}${data.length === 3000 ? " (3,000-row sample)" : ""}`}
      />
      <ResponsiveContainer width="100%" height={440}>
        <ScatterChart margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
          <CartesianGrid stroke="var(--grid)" />
          <XAxis
            dataKey="x"
            type="number"
            name={label(x)}
            tick={{ fill: "var(--muted)", fontSize: 12 }}
            stroke="var(--baseline)"
            label={{ value: label(x), position: "insideBottom", offset: -4, fill: "var(--ink-2)", fontSize: 12 }}
          />
          <YAxis
            dataKey="y"
            type="number"
            name={label(y)}
            tick={{ fill: "var(--muted)", fontSize: 12 }}
            stroke="var(--baseline)"
            label={{ value: label(y), angle: -90, position: "insideLeft", fill: "var(--ink-2)", fontSize: 12 }}
          />
          <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "3 3" }} />
          <Scatter data={data} fill="var(--series-1)" fillOpacity={0.55} />
        </ScatterChart>
      </ResponsiveContainer>
    </>
  );
}

/* ---------- Heatmap (CSS grid) ---------- */

interface HeatRow {
  x: string;
  y: string;
  count: number;
}

function Heatmap({ data, x, y }: { data: HeatRow[]; x: string; y: string }) {
  const xs = Array.from(new Set(data.map((d) => d.x)));
  const ys = Array.from(new Set(data.map((d) => d.y)));
  const lookup = new Map(data.map((d) => [`${d.x} ${d.y}`, d.count]));
  const max = Math.max(...data.map((d) => d.count), 1);

  const color = (n: number) => {
    if (n === 0) return "transparent";
    const t = Math.log(n + 1) / Math.log(max + 1);
    const idx = Math.min(SEQ.length - 1, Math.floor(t * SEQ.length));
    return SEQ[idx];
  };
  const inkFor = (n: number) => {
    const t = Math.log(n + 1) / Math.log(max + 1);
    return t >= 0.6 ? "#ffffff" : "var(--ink)";
  };

  return (
    <>
      <ChartTitle text={`${label(x)} × ${label(y)} — record counts (log color scale)`} />
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "separate", borderSpacing: 2, fontSize: 12 }}>
          <thead>
            <tr>
              <th />
              {ys.map((yv) => (
                <th
                  key={yv}
                  style={{
                    color: "var(--ink-2)",
                    fontWeight: 500,
                    padding: "4px 6px",
                    maxWidth: 90,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                  title={yv}
                >
                  {yv}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {xs.map((xv) => (
              <tr key={xv}>
                <th
                  style={{
                    color: "var(--ink-2)",
                    fontWeight: 500,
                    textAlign: "right",
                    padding: "2px 8px",
                    maxWidth: 190,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                  title={xv}
                >
                  {xv}
                </th>
                {ys.map((yv) => {
                  const n = lookup.get(`${xv} ${yv}`) ?? 0;
                  return (
                    <td
                      key={yv}
                      title={`${xv} × ${yv}: ${fmtInt(n)} records`}
                      className="tabular"
                      style={{
                        background: color(n),
                        color: inkFor(n),
                        textAlign: "center",
                        minWidth: 44,
                        padding: "5px 6px",
                        borderRadius: 4,
                        border: n === 0 ? "1px solid var(--grid)" : "1px solid transparent",
                      }}
                    >
                      {n > 0 ? fmtInt(n) : ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ChartTitle({ text }: { text: string }) {
  return (
    <h3 className="mb-3 text-sm font-semibold" style={{ color: "var(--ink)" }}>
      {text}
    </h3>
  );
}
