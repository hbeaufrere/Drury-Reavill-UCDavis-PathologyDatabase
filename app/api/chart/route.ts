import { NextRequest, NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { dbConfigured, query } from "@/lib/db";
import {
  NUMERIC_COLUMNS,
  buildWhere,
  parseFilters,
  searchableColumns,
} from "@/lib/filters";

export const dynamic = "force-dynamic";

const isNumeric = (c: string | null): c is string =>
  !!c && (NUMERIC_COLUMNS as readonly string[]).includes(c);

export async function GET(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ error: "DATABASE_URL is not configured" }, { status: 503 });
  }
  const admin = isAdmin(req);
  const textCols = searchableColumns(admin);
  const isText = (c: string | null): c is string => !!c && textCols.includes(c);
  const sp = req.nextUrl.searchParams;
  const filters = parseFilters(sp, admin);
  const where = buildWhere(filters);
  const type = sp.get("type");
  const topN = Math.min(50, Math.max(2, Number(sp.get("topN")) || 20));

  try {
    switch (type) {
      case "count": {
        // Bar and pie charts: category counts for one text column.
        const x = sp.get("x");
        if (!isText(x)) return badRequest("x must be a text column");
        const rows = await query<{ value: string | null; n: string }>(
          `SELECT ${x} AS value, count(*)::text AS n
           FROM records WHERE ${where.sql} AND ${x} IS NOT NULL
           GROUP BY ${x} ORDER BY count(*) DESC, ${x} LIMIT ${topN}`,
          where.params
        );
        return NextResponse.json({
          data: rows.map((r) => ({ value: r.value, count: Number(r.n) })),
        });
      }

      case "histogram": {
        const x = sp.get("x");
        const bins = Math.min(200, Math.max(5, Number(sp.get("bins")) || 40));
        if (!isNumeric(x)) return badRequest("x must be a numeric column");
        const stats = await query<{ lo: number | null; hi: number | null }>(
          `SELECT min(${x})::float8 AS lo, max(${x})::float8 AS hi
           FROM records WHERE ${where.sql} AND ${x} IS NOT NULL`,
          where.params
        );
        const { lo, hi } = stats[0];
        if (lo === null || hi === null) return NextResponse.json({ data: [] });
        if (lo === hi) {
          return NextResponse.json({ data: [{ x0: lo, x1: hi, count: 1 }] });
        }
        const rows = await query<{ bucket: number; n: string }>(
          `SELECT width_bucket(${x}, $${where.params.length + 1}, $${where.params.length + 2}, ${bins}) AS bucket,
                  count(*)::text AS n
           FROM records WHERE ${where.sql} AND ${x} IS NOT NULL
           GROUP BY bucket ORDER BY bucket`,
          [...where.params, lo, hi]
        );
        const step = (hi - lo) / bins;
        const data = rows.map((r) => {
          const b = Math.min(bins, Math.max(1, Number(r.bucket)));
          return {
            x0: lo + (b - 1) * step,
            x1: lo + b * step,
            count: Number(r.n),
          };
        });
        return NextResponse.json({ data, lo, hi, step });
      }

      case "box": {
        const x = sp.get("x");
        const y = sp.get("y");
        if (!isText(x)) return badRequest("x must be a text column");
        if (!isNumeric(y)) return badRequest("y must be a numeric column");
        const rows = await query<{
          group: string;
          n: string;
          min: number; q1: number; median: number; q3: number; max: number;
        }>(
          `WITH top_groups AS (
             SELECT ${x} AS g FROM records
             WHERE ${where.sql} AND ${x} IS NOT NULL AND ${y} IS NOT NULL
             GROUP BY ${x} ORDER BY count(*) DESC LIMIT ${topN}
           )
           SELECT ${x} AS group, count(*)::text AS n,
                  min(${y})::float8 AS min,
                  percentile_cont(0.25) WITHIN GROUP (ORDER BY ${y})::float8 AS q1,
                  percentile_cont(0.5)  WITHIN GROUP (ORDER BY ${y})::float8 AS median,
                  percentile_cont(0.75) WITHIN GROUP (ORDER BY ${y})::float8 AS q3,
                  max(${y})::float8 AS max
           FROM records
           WHERE ${where.sql} AND ${x} IN (SELECT g FROM top_groups) AND ${y} IS NOT NULL
           GROUP BY ${x} ORDER BY count(*) DESC`,
          where.params
        );
        return NextResponse.json({
          data: rows.map((r) => ({ ...r, n: Number(r.n) })),
        });
      }

      case "scatter": {
        const x = sp.get("x");
        const y = sp.get("y");
        if (!isNumeric(x) || !isNumeric(y)) {
          return badRequest("x and y must be numeric columns");
        }
        const rows = await query<{ x: number; y: number }>(
          `SELECT ${x}::float8 AS x, ${y}::float8 AS y
           FROM records
           WHERE ${where.sql} AND ${x} IS NOT NULL AND ${y} IS NOT NULL
           ORDER BY md5(id::text) LIMIT 3000`,
          where.params
        );
        return NextResponse.json({ data: rows });
      }

      case "heatmap": {
        const x = sp.get("x");
        const y = sp.get("y");
        if (!isText(x) || !isText(y) || x === y) {
          return badRequest("x and y must be distinct text columns");
        }
        const limit = Math.min(30, topN);
        const rows = await query<{ xv: string; yv: string; n: string }>(
          `WITH tx AS (
             SELECT ${x} AS v FROM records WHERE ${where.sql} AND ${x} IS NOT NULL
             GROUP BY ${x} ORDER BY count(*) DESC LIMIT ${limit}
           ), ty AS (
             SELECT ${y} AS v FROM records WHERE ${where.sql} AND ${y} IS NOT NULL
             GROUP BY ${y} ORDER BY count(*) DESC LIMIT ${limit}
           )
           SELECT ${x} AS xv, ${y} AS yv, count(*)::text AS n
           FROM records
           WHERE ${where.sql}
             AND ${x} IN (SELECT v FROM tx) AND ${y} IN (SELECT v FROM ty)
           GROUP BY ${x}, ${y}`,
          where.params
        );
        return NextResponse.json({
          data: rows.map((r) => ({ x: r.xv, y: r.yv, count: Number(r.n) })),
        });
      }

      default:
        return badRequest("unknown chart type");
    }
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "query failed" },
      { status: 500 }
    );
  }
}

function badRequest(msg: string) {
  return NextResponse.json({ error: msg }, { status: 400 });
}
