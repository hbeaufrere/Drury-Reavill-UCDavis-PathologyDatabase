import { randomBytes } from "node:crypto";
import { query } from "@/lib/db";

// Download access codes: issued by the admin (in response to email requests)
// and entered by visitors in the explorer to unlock CSV exports larger than
// the public limit. Codes never reveal patient names — only the admin
// session does that. The table is created on first use and survives
// re-seeding (the seed script only rebuilds `records`).

export interface DownloadCode {
  code: string;
  note: string | null;
  created_at: string;
  expires_at: string | null;
  revoked: boolean;
}

let ensured = false;

export async function ensureCodesTable(): Promise<void> {
  if (ensured) return;
  await query(`
    CREATE TABLE IF NOT EXISTS download_codes (
      code       text PRIMARY KEY,
      note       text,
      created_at timestamptz NOT NULL DEFAULT now(),
      expires_at timestamptz,
      revoked    boolean NOT NULL DEFAULT false
    )
  `);
  ensured = true;
}

export async function isValidCode(code: string): Promise<boolean> {
  const c = code.trim();
  if (!c) return false;
  await ensureCodesTable();
  const rows = await query(
    `SELECT 1 FROM download_codes
     WHERE code = $1 AND NOT revoked
       AND (expires_at IS NULL OR expires_at > now())`,
    [c]
  );
  return rows.length > 0;
}

export async function listCodes(): Promise<DownloadCode[]> {
  await ensureCodesTable();
  return query<DownloadCode & Record<string, unknown>>(
    `SELECT code, note, created_at::text, expires_at::text, revoked
     FROM download_codes ORDER BY created_at DESC`
  ) as Promise<DownloadCode[]>;
}

export async function createCode(
  note: string,
  expiresDays: number | null
): Promise<DownloadCode> {
  await ensureCodesTable();
  const code = randomBytes(8).toString("hex").toUpperCase();
  const rows = await query<DownloadCode & Record<string, unknown>>(
    `INSERT INTO download_codes (code, note, expires_at)
     VALUES ($1, $2, CASE WHEN $3::int IS NULL THEN NULL
                          ELSE now() + ($3::int || ' days')::interval END)
     RETURNING code, note, created_at::text, expires_at::text, revoked`,
    [code, note || null, expiresDays]
  );
  return rows[0] as DownloadCode;
}

export async function revokeCode(code: string): Promise<void> {
  await ensureCodesTable();
  await query(`UPDATE download_codes SET revoked = true WHERE code = $1`, [code]);
}
