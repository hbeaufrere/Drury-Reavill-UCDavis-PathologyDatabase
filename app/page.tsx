import Link from "next/link";
import { dbConfigured, query } from "@/lib/db";
import { fmtInt } from "@/lib/format";

export const revalidate = 3600;

interface Stats {
  total: number;
  mainTotal: number;
  cytoTotal: number;
  taxa: number;
  species: number;
  tumors: number;
  topTaxa: { value: string; n: number }[];
  topSpecies: { value: string; n: number }[];
}

async function loadStats(): Promise<Stats | null> {
  if (!dbConfigured()) return null;
  try {
    const [totals, distincts, tumors, topTaxa, topSpecies] = await Promise.all([
      query<{ dataset: string; n: string }>(
        `SELECT dataset, count(*)::text AS n FROM records GROUP BY dataset`
      ),
      query<{ taxa: string; species: string }>(
        `SELECT count(DISTINCT category)::text AS taxa,
                count(DISTINCT breed)::text AS species
         FROM records`
      ),
      query<{ n: string }>(
        `SELECT count(*)::text AS n FROM records WHERE specific_lesions = 'Tumor'`
      ),
      query<{ value: string; n: string }>(
        `SELECT category AS value, count(*)::text AS n FROM records
         WHERE category IS NOT NULL
         GROUP BY category ORDER BY count(*) DESC LIMIT 12`
      ),
      query<{ value: string; n: string }>(
        `SELECT breed AS value, count(*)::text AS n FROM records
         WHERE breed IS NOT NULL
         GROUP BY breed ORDER BY count(*) DESC LIMIT 10`
      ),
    ]);
    const byDs = Object.fromEntries(totals.map((t) => [t.dataset, Number(t.n)]));
    const mainTotal = byDs.main ?? 0;
    const cytoTotal = byDs.cytology ?? 0;
    return {
      total: mainTotal + cytoTotal,
      mainTotal,
      cytoTotal,
      taxa: Number(distincts[0].taxa),
      species: Number(distincts[0].species),
      tumors: Number(tumors[0].n),
      topTaxa: topTaxa.map((r) => ({ value: r.value, n: Number(r.n) })),
      topSpecies: topSpecies.map((r) => ({ value: r.value, n: Number(r.n) })),
    };
  } catch {
    return null;
  }
}

function StatTile({ value, caption }: { value: string; caption: string }) {
  return (
    <div className="card px-6 py-5">
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
    <div className="mx-auto max-w-5xl px-4 py-12 lg:px-8">
      <header className="mb-10 text-center">
        <p
          className="mb-3 text-xs font-semibold uppercase tracking-widest"
          style={{ color: "var(--accent)" }}
        >
          Exotic companion animal pathology
        </p>
        <h1 className="mx-auto max-w-3xl text-4xl font-bold tracking-tight lg:text-5xl">
          Drury R. Reavill Pathology Database at UC Davis
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg" style={{ color: "var(--ink-2)" }}>
          Decades of biopsy, necropsy, and cytology records from exotic
          companion animals — birds, reptiles, small mammals, amphibians, and
          fish — searchable in one place.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          <Link
            href="/explore"
            className="btn btn-primary px-6 py-3 text-base font-semibold"
          >
            Search the database →
          </Link>
        </div>
      </header>

      {!stats ? (
        <div className="card mx-auto max-w-lg p-8 text-center">
          <h2 className="mb-2 text-lg font-semibold">Database not available</h2>
          <p className="text-sm" style={{ color: "var(--ink-2)" }}>
            Set <code>DATABASE_URL</code> to a Neon Postgres connection string
            and run <code>npm run db:seed</code>. See the README for setup
            steps.
          </p>
        </div>
      ) : (
        <>
          <section
            aria-label="Headline statistics"
            className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6"
          >
            <StatTile value={fmtInt(stats.total)} caption="Case records" />
            <StatTile value={fmtInt(stats.mainTotal)} caption="Pathology reports" />
            <StatTile value={fmtInt(stats.cytoTotal)} caption="Cytology cases" />
            <StatTile value={fmtInt(stats.taxa)} caption="Taxa represented" />
            <StatTile value={fmtInt(stats.species)} caption="Species &amp; breeds" />
            <StatTile value={fmtInt(stats.tumors)} caption="Tumor diagnoses" />
          </section>

          <section className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-2">
            <div className="card p-6">
              <h2 className="mb-1 text-sm font-semibold">Records by taxon</h2>
              <p className="mb-4 text-xs" style={{ color: "var(--muted)" }}>
                Top {stats.topTaxa.length} taxonomic groups by case count
              </p>
              <ul className="flex flex-col gap-2.5">
                {stats.topTaxa.map((t) => {
                  const pct = (t.n / stats.total) * 100;
                  const barPct = (t.n / stats.topTaxa[0].n) * 100;
                  return (
                    <li key={t.value} className="text-sm">
                      <div className="mb-1 flex items-baseline justify-between gap-2">
                        <span className="truncate">{t.value}</span>
                        <span
                          className="tabular shrink-0 text-xs"
                          style={{ color: "var(--ink-2)" }}
                        >
                          {fmtInt(t.n)} · {pct.toFixed(1)}%
                        </span>
                      </div>
                      <div
                        aria-hidden
                        style={{
                          height: 8,
                          borderRadius: 4,
                          background: "var(--grid)",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${barPct}%`,
                            height: "100%",
                            borderRadius: 4,
                            background: "var(--series-1)",
                          }}
                        />
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>

            <div className="card p-6">
              <h2 className="mb-1 text-sm font-semibold">Most represented species</h2>
              <p className="mb-4 text-xs" style={{ color: "var(--muted)" }}>
                Top {stats.topSpecies.length} species and breeds by case count
              </p>
              <ol className="flex flex-col">
                {stats.topSpecies.map((s, i) => (
                  <li
                    key={s.value}
                    className="flex items-center gap-3 border-b py-2.5 text-sm last:border-0"
                    style={{ borderColor: "var(--grid)" }}
                  >
                    <span
                      className="tabular w-6 text-right text-xs font-semibold"
                      style={{ color: "var(--muted)" }}
                    >
                      {i + 1}
                    </span>
                    <span className="flex-1 truncate">{s.value}</span>
                    <span className="tabular text-xs" style={{ color: "var(--ink-2)" }}>
                      {fmtInt(s.n)}
                    </span>
                  </li>
                ))}
              </ol>
              <p className="mt-4 text-xs" style={{ color: "var(--muted)" }}>
                Explore the full species list — and every diagnosis — in the
                database search.
              </p>
            </div>
          </section>

          <section className="mt-10 text-center">
            <Link
              href="/explore"
              className="btn btn-primary px-6 py-3 text-base font-semibold"
            >
              Search {fmtInt(stats.total)} records →
            </Link>
            <p className="mt-3 text-sm" style={{ color: "var(--muted)" }}>
              Full-text diagnosis search · filters by taxon, species, sex, age,
              body system, and disease process · charts · CSV export
            </p>
          </section>
        </>
      )}

      <footer
        className="mt-14 border-t pt-5 text-center text-xs"
        style={{ borderColor: "var(--grid)", color: "var(--muted)" }}
      >
        Patient identities are anonymized for public access.{" "}
        <Link href="/admin" style={{ color: "var(--accent)" }}>
          Admin sign-in
        </Link>
      </footer>
    </div>
  );
}
