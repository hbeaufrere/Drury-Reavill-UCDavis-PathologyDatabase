import { NextRequest, NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { dbConfigured } from "@/lib/db";
import { getVisitStats } from "@/lib/visits";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ error: "DATABASE_URL is not configured" }, { status: 503 });
  }
  if (!isAdmin(req)) {
    return NextResponse.json({ error: "Admin access required" }, { status: 401 });
  }
  return NextResponse.json(await getVisitStats());
}
