import { NextRequest, NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { dbConfigured, query } from "@/lib/db";
import { buildWhere, parseFilters, visibleColumns } from "@/lib/filters";

export const dynamic = "force-dynamic";

const MAX_PAGE_SIZE = 200;

export async function GET(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ error: "DATABASE_URL is not configured" }, { status: 503 });
  }
  const admin = isAdmin(req);
  const cols = visibleColumns(admin);
  const sp = req.nextUrl.searchParams;
  const filters = parseFilters(sp, admin);
  const where = buildWhere(filters);

  const page = Math.max(1, Number(sp.get("page")) || 1);
  const pageSize = Math.min(MAX_PAGE_SIZE, Math.max(1, Number(sp.get("pageSize")) || 50));

  const sortCol = sp.get("sort");
  const sortDir = sp.get("dir") === "desc" ? "DESC" : "ASC";
  const orderBy =
    sortCol && cols.includes(sortCol)
      ? `${sortCol} ${sortDir} NULLS LAST, id`
      : "id";

  const countP = query<{ n: string }>(
    `SELECT count(*)::text AS n FROM records WHERE ${where.sql}`,
    where.params
  );
  const rowsP = query(
    `SELECT id, ${cols.join(", ")} FROM records
     WHERE ${where.sql}
     ORDER BY ${orderBy}
     LIMIT ${pageSize} OFFSET ${(page - 1) * pageSize}`,
    where.params
  );
  const [count, rows] = await Promise.all([countP, rowsP]);

  return NextResponse.json({
    total: Number(count[0].n),
    page,
    pageSize,
    rows,
  });
}
