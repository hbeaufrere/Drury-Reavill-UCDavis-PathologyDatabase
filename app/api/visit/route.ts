import { NextRequest, NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { dbConfigured } from "@/lib/db";
import { recordVisit } from "@/lib/visits";

export const dynamic = "force-dynamic";

// Beaconed once per browser session by VisitTracker. Admin sessions and
// obvious bots are not counted. On Vercel the visitor's country arrives in
// the x-vercel-ip-country header (ISO 3166-1 alpha-2).

const BOT_RE = /bot|crawl|spider|slurp|preview|fetch|monitor|curl|wget|python|headless/i;

export async function POST(req: NextRequest) {
  if (!dbConfigured() || isAdmin(req)) {
    return new NextResponse(null, { status: 204 });
  }
  const ua = req.headers.get("user-agent") ?? "";
  if (BOT_RE.test(ua)) {
    return new NextResponse(null, { status: 204 });
  }
  const country = (req.headers.get("x-vercel-ip-country") ?? "").trim().toUpperCase();
  const code = /^[A-Z]{2}$/.test(country) ? country : "Unknown";
  try {
    await recordVisit(code);
  } catch {
    // Counting must never break the site.
  }
  return new NextResponse(null, { status: 204 });
}
