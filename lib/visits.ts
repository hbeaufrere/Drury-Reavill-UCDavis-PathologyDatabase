import { query } from "@/lib/db";

// Visit counting: one visit per browser session (the client beacons once per
// session), aggregated as daily per-country counters so no per-visitor data
// is ever stored. Admin sessions are never counted. The table is created on
// first use and survives re-seeding (the seed script only rebuilds `records`).

let ensured = false;

export async function ensureVisitsTable(): Promise<void> {
  if (ensured) return;
  await query(`
    CREATE TABLE IF NOT EXISTS visits (
      day     date NOT NULL,
      country text NOT NULL,
      n       integer NOT NULL DEFAULT 0,
      PRIMARY KEY (day, country)
    )
  `);
  ensured = true;
}

export async function recordVisit(country: string): Promise<void> {
  await ensureVisitsTable();
  await query(
    `INSERT INTO visits (day, country, n) VALUES (CURRENT_DATE, $1, 1)
     ON CONFLICT (day, country) DO UPDATE SET n = visits.n + 1`,
    [country]
  );
}

export interface VisitStats {
  total: number;
  byYear: { year: number; n: number }[];
  byCountry: { country: string; total: number; years: Record<string, number> }[];
}

export async function getVisitStats(): Promise<VisitStats> {
  await ensureVisitsTable();
  const rows = await query<{ year: number; country: string; n: string }>(
    `SELECT EXTRACT(YEAR FROM day)::int AS year, country, sum(n)::text AS n
     FROM visits GROUP BY 1, 2 ORDER BY 1, 2`
  );
  let total = 0;
  const byYearMap = new Map<number, number>();
  const byCountryMap = new Map<string, { total: number; years: Record<string, number> }>();
  for (const r of rows) {
    const n = Number(r.n);
    total += n;
    byYearMap.set(r.year, (byYearMap.get(r.year) ?? 0) + n);
    if (!byCountryMap.has(r.country)) {
      byCountryMap.set(r.country, { total: 0, years: {} });
    }
    const c = byCountryMap.get(r.country)!;
    c.total += n;
    c.years[String(r.year)] = (c.years[String(r.year)] ?? 0) + n;
  }
  return {
    total,
    byYear: [...byYearMap.entries()]
      .map(([year, n]) => ({ year, n }))
      .sort((a, b) => a.year - b.year),
    byCountry: [...byCountryMap.entries()]
      .map(([country, v]) => ({ country, ...v }))
      .sort((a, b) => b.total - a.total),
  };
}
