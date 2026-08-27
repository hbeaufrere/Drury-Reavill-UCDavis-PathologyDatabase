import { NextRequest, NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { dbConfigured, query } from "@/lib/db";
import { DATASETS, Dataset, NUMERIC_COLUMNS, visibleColumns } from "@/lib/filters";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ error: "DATABASE_URL is not configured" }, { status: 503 });
  }
  const ALL_COLUMNS = visibleColumns(isAdmin(req));
  const ds = req.nextUrl.searchParams.get("dataset");
  const dataset: Dataset = DATASETS.includes(ds as Dataset) ? (ds as Dataset) : "main";

  const pieces = ALL_COLUMNS.map(
    (c) =>
      `count(${c})::text AS "${c}__nonnull", count(DISTINCT ${c})::text AS "${c}__distinct"`
  ).join(", ");
  const numPieces = NUMERIC_COLUMNS.map(
    (c) =>
      `min(${c})::float8 AS "${c}__min", max(${c})::float8 AS "${c}__max",
       avg(${c})::float8 AS "${c}__avg",
       percentile_cont(0.5) WITHIN GROUP (ORDER BY ${c})::float8 AS "${c}__median"`
  ).join(", ");

  const rows = await query(
    `SELECT count(*)::text AS total, ${pieces}, ${numPieces}
     FROM records WHERE dataset = $1`,
    [dataset]
  );
  const r = rows[0] as Record<string, unknown>;

  const columns = ALL_COLUMNS.map((c) => ({
    name: c,
    nonNull: Number(r[`${c}__nonnull`]),
    distinct: Number(r[`${c}__distinct`]),
    numeric: (NUMERIC_COLUMNS as readonly string[]).includes(c)
      ? {
          min: r[`${c}__min`] as number | null,
          max: r[`${c}__max`] as number | null,
          avg: r[`${c}__avg`] as number | null,
          median: r[`${c}__median`] as number | null,
        }
      : null,
  }));

  return NextResponse.json({ dataset, total: Number(r.total), columns });
}
