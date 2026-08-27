import { createHash, timingSafeEqual } from "node:crypto";
import type { NextRequest } from "next/server";

// Patient names are stored in the database but only exposed to the admin.
// Admin access is a single shared password (ADMIN_PASSWORD env var); a
// successful login sets an httpOnly cookie holding a hash of the password,
// which every API route verifies statelessly.

export const ADMIN_COOKIE = "drury_admin";

export function adminToken(): string | null {
  const pw = process.env.ADMIN_PASSWORD;
  if (!pw) return null;
  return createHash("sha256").update(`drury-admin:${pw}`).digest("hex");
}

export function isAdmin(req: NextRequest): boolean {
  const expected = adminToken();
  if (!expected) return false;
  const got = req.cookies.get(ADMIN_COOKIE)?.value ?? "";
  if (got.length !== expected.length) return false;
  return timingSafeEqual(Buffer.from(got), Buffer.from(expected));
}

export function checkPassword(password: string): boolean {
  const pw = process.env.ADMIN_PASSWORD;
  if (!pw) return false;
  const a = createHash("sha256").update(password).digest();
  const b = createHash("sha256").update(pw).digest();
  return timingSafeEqual(a, b);
}
