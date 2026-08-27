import { neon } from "@neondatabase/serverless";
import { Pool } from "pg";

// Neon's HTTP driver is used when the connection string points at Neon
// (best cold-start behavior on Vercel functions); a regular pg Pool is
// used everywhere else (local development, tests, other Postgres hosts).

export type Row = Record<string, unknown>;

const url = process.env.DATABASE_URL ?? "";
const isNeon = /neon\.tech/.test(url);

let pool: Pool | null = null;
function getPool(): Pool {
  if (!pool) pool = new Pool({ connectionString: url, max: 5 });
  return pool;
}

export function dbConfigured(): boolean {
  return url.length > 0;
}

export async function query<T extends Row = Row>(
  text: string,
  params: unknown[] = []
): Promise<T[]> {
  if (!url) throw new Error("DATABASE_URL is not set");
  if (isNeon) {
    const sql = neon(url);
    return (await sql.query(text, params)) as T[];
  }
  const res = await getPool().query(text, params);
  return res.rows as T[];
}
