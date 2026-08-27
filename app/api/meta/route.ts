import { NextRequest, NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { dbConfigured, query } from "@/lib/db";
import { DATASETS, Dataset, FACET_COLUMNS } from "@/lib/filters";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ error: "DATABASE_URL is not configured" }, { status: 503 });
  }
  const ds = req.nextUrl.searchParams.get("dataset");
  const dataset: Dataset = DATASETS.includes(ds as Dataset) ? (ds as Dataset) : "main";

  const [totalRows, ageRows, ...facetRows] = await Promise.all([
    query<{ n: string }>(`SELECT count(*)::text AS n FROM records WHERE dataset = $1`, [dataset]),
    query<{ min: number | null; max: number | null }>(
      `SELECT min(age) AS min, max(age) AS max FROM records WHERE dataset = $1`,
      [dataset]
    ),
    ...FACET_COLUMNS.map((col) =>
      query<{ value: string; n: string }>(
        `SELECT ${col} AS value, count(*)::text AS n
         FROM records WHERE dataset = $1 AND ${col} IS NOT NULL
         GROUP BY ${col} ORDER BY count(*) DESC, ${col}`,
        [dataset]
      )
    ),
  ]);

  const facets: Record<string, { value: string; count: number }[]> = {};
  FACET_COLUMNS.forEach((col, i) => {
    facets[col] = facetRows[i].map((r) => ({ value: r.value, count: Number(r.n) }));
  });

  return NextResponse.json({
    dataset,
    admin: isAdmin(req),
    total: Number(totalRows[0].n),
    ageRange: { min: ageRows[0].min, max: ageRows[0].max },
    facets,
  });
}
