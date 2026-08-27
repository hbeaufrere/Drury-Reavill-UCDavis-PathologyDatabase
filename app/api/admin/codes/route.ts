import { NextRequest, NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { dbConfigured } from "@/lib/db";
import { createCode, listCodes, revokeCode } from "@/lib/codes";

export const dynamic = "force-dynamic";

function guard(req: NextRequest): NextResponse | null {
  if (!dbConfigured()) {
    return NextResponse.json({ error: "DATABASE_URL is not configured" }, { status: 503 });
  }
  if (!isAdmin(req)) {
    return NextResponse.json({ error: "Admin access required" }, { status: 401 });
  }
  return null;
}

export async function GET(req: NextRequest) {
  const denied = guard(req);
  if (denied) return denied;
  return NextResponse.json({ codes: await listCodes() });
}

export async function POST(req: NextRequest) {
  const denied = guard(req);
  if (denied) return denied;
  let note = "";
  let expiresDays: number | null = null;
  try {
    const body = await req.json();
    note = String(body.note ?? "").slice(0, 200);
    const d = Number(body.expiresDays);
    expiresDays = Number.isFinite(d) && d > 0 ? Math.min(3650, Math.floor(d)) : null;
  } catch {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  }
  return NextResponse.json({ code: await createCode(note, expiresDays) });
}

export async function DELETE(req: NextRequest) {
  const denied = guard(req);
  if (denied) return denied;
  const code = req.nextUrl.searchParams.get("code") ?? "";
  if (!code) return NextResponse.json({ error: "code required" }, { status: 400 });
  await revokeCode(code);
  return NextResponse.json({ ok: true });
}
