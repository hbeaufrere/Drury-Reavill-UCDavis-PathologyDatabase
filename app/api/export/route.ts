import { NextRequest, NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { dbConfigured, query, type Row } from "@/lib/db";
import { buildWhere, parseFilters, visibleColumns } from "@/lib/filters";

export const dynamic = "force-dynamic";

const CHUNK = 5000;

function csvField(v: unknown): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export async function GET(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ error: "DATABASE_URL is not configured" }, { status: 503 });
  }
  const admin = isAdmin(req);
  const cols = visibleColumns(admin);
  const filters = parseFilters(req.nextUrl.searchParams, admin);
  const where = buildWhere(filters);
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        controller.enqueue(encoder.encode(cols.join(",") + "\n"));
        let lastId = 0;
        for (;;) {
          const p = [...where.params, lastId];
          const rows = await query<Row & { id: number }>(
            `SELECT id, ${cols.join(", ")} FROM records
             WHERE ${where.sql} AND id > $${p.length}
             ORDER BY id LIMIT ${CHUNK}`,
            p
          );
          if (!rows.length) break;
          const chunk = rows
            .map((r) => cols.map((c) => csvField(r[c])).join(","))
            .join("\n");
          controller.enqueue(encoder.encode(chunk + "\n"));
          lastId = Number(rows[rows.length - 1].id);
          if (rows.length < CHUNK) break;
        }
        controller.close();
      } catch (e) {
        controller.error(e);
      }
    },
  });

  return new NextResponse(stream, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="drury_${filters.dataset}_export.csv"`,
    },
  });
}
