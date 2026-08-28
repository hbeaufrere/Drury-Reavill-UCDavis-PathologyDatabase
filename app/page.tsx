import Link from "next/link";
import Citation from "@/components/Citation";
import { dbConfigured, query } from "@/lib/db";
import { fmtInt } from "@/lib/format";
import { CLASSES, TaxonClass, classify } from "@/lib/taxonomy";

export const revalidate = 3600;

const CLASS_COLORS: Record<TaxonClass, string> = {
  Birds: "var(--cls-1)",
  Mammals: "var(--cls-2)",
  Reptiles: "var(--cls-3)",
  Fish: "var(--cls-4)",
  Amphibians: "var(--cls-5)",
  Invertebrates: "var(--cls-6)",
};

interface OrderStat {
  order: string;
  n: number;
}
interface ClassStat {
  cls: TaxonClass;
  n: number;
  orders: OrderStat[];
}

interface Stats {
  total: number;
  mainTotal: number;
  cytoTotal: number;
  species: number;
  classes: ClassStat[];
  classified: number;
  unclassified: number;
  bodySystems: { value: string; n: number }[];
  processes: { value: string; n: number }[];
}

async function loadStats(): Promise<Stats | null> {
  if (!dbConfigured()) return null;
  try {
    const [totals, distincts, cats, systems, lesions] = await Promise.all([
      query<{ dataset: string; n: string }>(
        `SELECT dataset, count(*)::text AS n FROM records GROUP BY dataset`
      ),
      query<{ species: string }>(
        `SELECT count(DISTINCT breed)::text AS species FROM records`
      ),
      query<{ value: string; n: string }>(
        `SELECT category AS value, count(*)::text AS n FROM records
         WHERE category IS NOT NULL GROUP BY category`
      ),
      query<{ value: string; n: string }>(
        `SELECT diagnosis_category AS value, count(*)::text AS n FROM records
         WHERE diagnosis_category IS NOT NULL
         GROUP BY diagnosis_category ORDER BY count(*) DESC LIMIT 10`
      ),
      query<{ value: string; n: string }>(
        `SELECT CASE WHEN specific_lesions LIKE 'Infection%' THEN 'Infection'
                     ELSE specific_lesions END AS value,
                count(*)::text AS n
         FROM records WHERE specific_lesions IS NOT NULL
         GROUP BY 1 ORDER BY count(*) DESC LIMIT 8`
      ),
    ]);

    const byDs = Object.fromEntries(totals.map((t) => [t.dataset, Number(t.n)]));
    const mainTotal = byDs.main ?? 0;
    const cytoTotal = byDs.cytology ?? 0;
    const total = mainTotal + cytoTotal;

    // Aggregate raw categories into class -> order groups.
    const classMap = new Map<TaxonClass, Map<string, number>>();
    let classified = 0;
    for (const c of cats) {
      const info = classify(c.value);
      if (!info) continue;
      const n = Number(c.n);
      classified += n;
      if (!classMap.has(info.cls)) classMap.set(info.cls, new Map());
      const orders = classMap.get(info.cls)!;
      orders.set(info.order, (orders.get(info.order) ?? 0) + n);
    }
    const classes: ClassStat[] = CLASSES.map((cls) => {
      const orders = classMap.get(cls) ?? new Map<string, number>();
      const list = [...orders.entries()]
        .map(([order, n]) => ({ order, n }))
        .sort((a, b) => b.n - a.n);
      return { cls, n: list.reduce((s, o) => s + o.n, 0), orders: list };
    })
      .filter((c) => c.n > 0)
      .sort((a, b) => b.n - a.n);

    return {
      total,
      mainTotal,
      cytoTotal,
      species: Number(distincts[0].species),
      classes,
      classified,
      unclassified: total - classified,
      bodySystems: systems.map((r) => ({ value: r.value, n: Number(r.n) })),
      processes: lesions.map((r) => ({ value: r.value, n: Number(r.n) })),
    };
  } catch {
    return null;
  }
}

/* ---------- Server-rendered SVG donut ---------- */

function polar(cx: number, cy: number, r: number, angle: number) {
  return [cx + r * Math.cos(angle - Math.PI / 2), cy + r * Math.sin(angle - Math.PI / 2)];
}

function Donut({ classes, classified }: { classes: ClassStat[]; classified: number }) {
  const size = 300;
  const c = size / 2;
  const rOuter = 140;
  const rInner = 84;
  const tau = Math.PI * 2;
  let acc = 0;

  const slices = classes.map((cl) => {
    const frac = cl.n / classified;
    const a0 = acc * tau;
    acc += frac;
    const a1 = acc * tau;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const [x0o, y0o] = polar(c, c, rOuter, a0);
    const [x1o, y1o] = polar(c, c, rOuter, a1);
    const [x0i, y0i] = polar(c, c, rInner, a0);
    const [x1i, y1i] = polar(c, c, rInner, a1);
    const mid = (a0 + a1) / 2;
    const [lx, ly] = polar(c, c, (rOuter + rInner) / 2, mid);
    return {
      cls: cl.cls,
      frac,
      path: `M ${x0o} ${y0o} A ${rOuter} ${rOuter} 0 ${large} 1 ${x1o} ${y1o} L ${x1i} ${y1i} A ${rInner} ${rInner} 0 ${large} 0 ${x0i} ${y0i} Z`,
      lx,
      ly,
    };
  });

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      role="img"
      aria-label="Share of records by taxonomic class"
      style={{ maxWidth: "100%", height: "auto" }}
    >
      {slices.map((s) => (
        <path
          key={s.cls}
          d={s.path}
          fill={CLASS_COLORS[s.cls]}
          stroke="var(--surface)"
          strokeWidth="2"
        >
          <title>{`${s.cls}: ${(s.frac * 100).toFixed(1)}%`}</title>
        </path>
      ))}
      {slices
        .filter((s) => s.frac >= 0.05)
        .map((s) => (
          <text
            key={s.cls}
            x={s.lx}
            y={s.ly + 4}
            textAnchor="middle"
            fontSize="13"
            fontWeight="600"
            fill={s.cls === "Mammals" ? "#221a00" : "#ffffff"}
          >
            {(s.frac * 100).toFixed(0)}%
          </text>
        ))}
      <text x={c} y={c - 6} textAnchor="middle" fontSize="26" fontWeight="700" fill="var(--ink)">
        {fmtInt(classified)}
      </text>
      <text x={c} y={c + 16} textAnchor="middle" fontSize="12" fill="var(--muted)">
        classified records
      </text>
    </svg>
  );
}

function BarList({
  rows,
  color,
  total,
}: {
  rows: { label: string; n: number }[];
  color: string;
  total?: number;
}) {
  const max = rows[0]?.n ?? 1;
  return (
    <ul className="flex flex-col gap-2">
      {rows.map((r) => (
        <li key={r.label} className="text-sm">
          <div className="mb-0.5 flex items-baseline justify-between gap-2">
            <span className="truncate">{r.label}</span>
            <span className="tabular shrink-0 text-xs" style={{ color: "var(--ink-2)" }}>
              {fmtInt(r.n)}
              {total ? ` · ${((r.n / total) * 100).toFixed(1)}%` : ""}
            </span>
          </div>
          <div
            aria-hidden
            style={{ height: 7, borderRadius: 4, background: "var(--grid)", overflow: "hidden" }}
          >
            <div
              style={{
                width: `${Math.max(1.5, (r.n / max) * 100)}%`,
                height: "100%",
                borderRadius: 4,
                background: color,
              }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

function StatTile({ value, caption }: { value: string; caption: string }) {
  return (
    <div
      className="card px-6 py-5"
      style={{ borderTop: "3px solid var(--ucd-gold)" }}
    >
      <div className="text-3xl font-bold tracking-tight" style={{ color: "var(--ink)" }}>
        {value}
      </div>
      <div className="mt-1 text-sm" style={{ color: "var(--ink-2)" }}>
        {caption}
      </div>
    </div>
  );
}

export default async function LandingPage() {
  const stats = await loadStats();

  return (
    <div>
      {/* Hero — Aggie Blue in both themes */}
      <header style={{ background: "var(--ucd-blue)" }}>
        <div className="mx-auto max-w-6xl px-4 py-16 text-center lg:px-8">
          <p
            className="mb-3 text-xs font-bold uppercase tracking-widest"
            style={{ color: "var(--ucd-gold)" }}
          >
            Exotic companion animal pathology
          </p>
          <h1 className="mx-auto max-w-3xl text-4xl font-bold tracking-tight text-white lg:text-5xl">
            Drury R. Reavill Pathology Database at UC Davis
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-lg" style={{ color: "#c9d4e4" }}>
            Decades of biopsy, necropsy, and cytology records from exotic
            companion animals — birds, reptiles, small mammals, amphibians, and
            fish — searchable in one place.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/explore"
              className="inline-flex items-center gap-2 rounded-lg px-7 py-3 text-base font-bold"
              style={{ background: "var(--ucd-gold)", color: "var(--ucd-blue)" }}
            >
              Search the database →
            </Link>
            <Link
              href="/epidemiology"
              className="inline-flex items-center gap-2 rounded-lg border px-7 py-3 text-base font-bold text-white"
              style={{ borderColor: "rgba(255,255,255,0.45)" }}
            >
              Explore epidemiology of diseases in exotics
            </Link>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-4 pb-12 lg:px-8">
        {!stats ? (
          <div className="card mx-auto mt-12 max-w-lg p-8 text-center">
            <h2 className="mb-2 text-lg font-semibold">Database not available</h2>
            <p className="text-sm" style={{ color: "var(--ink-2)" }}>
              Set <code>DATABASE_URL</code> to a Neon Postgres connection string
              and run <code>npm run db:seed</code>. See the README for setup steps.
            </p>
          </div>
        ) : (
          <>
            <section
              aria-label="Headline statistics"
              className="mt-[-2rem] grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-4"
            >
              <StatTile value={fmtInt(stats.total)} caption="Case records" />
              <StatTile value={fmtInt(stats.mainTotal)} caption="Pathology reports" />
              <StatTile value={fmtInt(stats.cytoTotal)} caption="Cytology cases" />
              <StatTile value={fmtInt(stats.species)} caption="Species &amp; breeds" />
            </section>

            {/* Class-level donut + legend */}
            <section className="card mt-8 p-6 lg:p-8">
              <h2 className="text-base font-bold">Records by taxonomic class</h2>
              <p className="mb-6 mt-1 text-xs" style={{ color: "var(--muted)" }}>
                {fmtInt(stats.classified)} records with a recorded taxon;{" "}
                {fmtInt(stats.unclassified)} without one are not shown.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-10">
                <Donut classes={stats.classes} classified={stats.classified} />
                <ul className="flex min-w-56 flex-col gap-3">
                  {stats.classes.map((c) => (
                    <li key={c.cls} className="flex items-center gap-3 text-sm">
                      <span
                        aria-hidden
                        style={{
                          width: 12,
                          height: 12,
                          borderRadius: 3,
                          background: CLASS_COLORS[c.cls],
                          display: "inline-block",
                          flexShrink: 0,
                        }}
                      />
                      <span className="flex-1 font-medium">{c.cls}</span>
                      <span className="tabular" style={{ color: "var(--ink-2)" }}>
                        {fmtInt(c.n)}
                      </span>
                      <span
                        className="tabular w-14 text-right text-xs"
                        style={{ color: "var(--muted)" }}
                      >
                        {((c.n / stats.classified) * 100).toFixed(1)}%
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>

            {/* Orders within each class */}
            <section className="mt-8">
              <h2 className="mb-4 text-base font-bold">Orders within each class</h2>
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
                {stats.classes.map((c) => (
                  <div key={c.cls} className="card p-5">
                    <div className="mb-4 flex items-center gap-2">
                      <span
                        aria-hidden
                        style={{
                          width: 12,
                          height: 12,
                          borderRadius: 3,
                          background: CLASS_COLORS[c.cls],
                          display: "inline-block",
                        }}
                      />
                      <h3 className="flex-1 text-sm font-bold">{c.cls}</h3>
                      <span className="tabular text-xs" style={{ color: "var(--muted)" }}>
                        {fmtInt(c.n)} records
                      </span>
                    </div>
                    <BarList
                      rows={c.orders.slice(0, 8).map((o) => ({ label: o.order, n: o.n }))}
                      color={CLASS_COLORS[c.cls]}
                    />
                    {c.orders.length > 8 && (
                      <p className="mt-3 text-xs" style={{ color: "var(--muted)" }}>
                        + {c.orders.length - 8} more orders in the database
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {/* Body systems & disease processes */}
            <section className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-2">
              <div className="card p-6">
                <h2 className="mb-1 text-base font-bold">Body systems</h2>
                <p className="mb-4 text-xs" style={{ color: "var(--muted)" }}>
                  Diagnoses classified by organ system
                </p>
                <BarList
                  rows={stats.bodySystems.map((b) => ({ label: b.value, n: b.n }))}
                  color="var(--cls-1)"
                />
              </div>
              <div className="card p-6">
                <h2 className="mb-1 text-base font-bold">Disease processes</h2>
                <p className="mb-4 text-xs" style={{ color: "var(--muted)" }}>
                  Tumors, infections, and other classified processes
                </p>
                <BarList
                  rows={stats.processes.map((p) => ({ label: p.value, n: p.n }))}
                  color="var(--cls-3)"
                />
              </div>
            </section>

            <section className="mt-10 text-center">
              <div className="flex flex-wrap items-center justify-center gap-3">
                <Link
                  href="/explore"
                  className="inline-flex items-center gap-2 rounded-lg px-7 py-3 text-base font-bold"
                  style={{ background: "var(--ucd-blue)", color: "#ffffff" }}
                >
                  Search {fmtInt(stats.total)} records →
                </Link>
                <Link
                  href="/epidemiology"
                  className="inline-flex items-center gap-2 rounded-lg border px-7 py-3 text-base font-bold"
                  style={{ borderColor: "var(--ucd-blue)", color: "var(--ink)" }}
                >
                  Explore disease epidemiology
                </Link>
              </div>
              <p className="mt-3 text-sm" style={{ color: "var(--muted)" }}>
                Full-text diagnosis search · filters by taxon, species, sex, age,
                organ count, body system, and disease process · charts · CSV export
              </p>
            </section>
          </>
        )}

        {/* Citation & data use */}
        <section id="cite" className="mt-14 grid grid-cols-1 gap-5 lg:grid-cols-2">
          <div>
            <h2 className="mb-3 text-base font-bold">How to cite this database</h2>
            <Citation />
          </div>
          <div>
            <h2 className="mb-3 text-base font-bold">Data use</h2>
            <p className="text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>
              Search results under 1,000 records can be downloaded as CSV
              directly from the explorer. For larger extracts or research
              collaborations, email{" "}
              <a
                href="mailto:hbeaufrere@ucdavis.edu"
                className="font-medium"
                style={{ color: "var(--accent)" }}
              >
                hbeaufrere@ucdavis.edu
              </a>{" "}
              describing the purpose of your request and how the data will be
              used — approved requests receive a download access code that
              unlocks larger exports in the explorer. Patient identities are
              anonymized for public access.
            </p>
          </div>
        </section>

        <footer
          className="mt-12 border-t pt-5 text-center text-xs"
          style={{ borderColor: "var(--grid)", color: "var(--muted)" }}
        >
          Drury R. Reavill Pathology Database · University of California, Davis ·{" "}
          <Link href="/admin" style={{ color: "var(--accent)" }}>
            Admin sign-in
          </Link>
        </footer>
      </div>
    </div>
  );
}
