#!/usr/bin/env node
// Load data/*.ndjson.gz into the Postgres database at DATABASE_URL.
// Usage: DATABASE_URL=postgres://... npm run db:seed
// Re-runnable: drops and recreates the records table.

import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import { createGunzip } from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const FILES = ["main.ndjson.gz", "cytology.ndjson.gz"];

const url = process.env.DATABASE_URL;
if (!url) {
  console.error("DATABASE_URL is not set. Use the Neon *direct* (non-pooled) connection string for seeding.");
  process.exit(1);
}

const COLS = [
  "dataset", "animal_name", "category", "breed", "sex", "age", "age_text",
  "diagnosis", "tissues", "stains", "stains_charge", "charge_type",
  "diagnosis_category", "specific_lesions",
];

async function* readRows(file) {
  const rl = createInterface({
    input: createReadStream(file).pipe(createGunzip()),
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    if (line.trim()) yield JSON.parse(line);
  }
}

const client = new pg.Client({ connectionString: url });
await client.connect();

console.log("Creating schema...");
await client.query(`
  CREATE EXTENSION IF NOT EXISTS pg_trgm;
  DROP TABLE IF EXISTS records;
  CREATE TABLE records (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset            text NOT NULL,
    animal_name        text,
    category           text,
    breed              text,
    sex                text,
    age                real,
    age_text           text,
    diagnosis          text,
    tissues            integer,
    stains             text,
    stains_charge      real,
    charge_type        text,
    diagnosis_category text,
    specific_lesions   text
  );
`);

let grandTotal = 0;
for (const name of FILES) {
  const file = path.join(ROOT, "data", name);
  let batch = [];
  let count = 0;

  async function flush() {
    if (!batch.length) return;
    const values = [];
    const params = [];
    let p = 1;
    for (const row of batch) {
      const placeholders = COLS.map(() => `$${p++}`);
      values.push(`(${placeholders.join(",")})`);
      for (const c of COLS) params.push(row[c] ?? null);
    }
    await client.query(
      `INSERT INTO records (${COLS.join(",")}) VALUES ${values.join(",")}`,
      params
    );
    count += batch.length;
    batch = [];
  }

  for await (const row of readRows(file)) {
    batch.push(row);
    if (batch.length >= 500) await flush();
  }
  await flush();
  grandTotal += count;
  console.log(`${name}: ${count} rows`);
}

console.log("Creating indexes...");
await client.query(`
  CREATE INDEX records_dataset_idx ON records (dataset);
  CREATE INDEX records_category_idx ON records (dataset, category);
  CREATE INDEX records_sex_idx ON records (dataset, sex);
  CREATE INDEX records_breed_idx ON records (dataset, breed);
  CREATE INDEX records_diag_cat_idx ON records (dataset, diagnosis_category);
  CREATE INDEX records_lesions_idx ON records (dataset, specific_lesions);
  CREATE INDEX records_age_idx ON records (dataset, age);
  CREATE INDEX records_diagnosis_trgm ON records USING gin (diagnosis gin_trgm_ops);
  CREATE INDEX records_breed_trgm ON records USING gin (breed gin_trgm_ops);
  ANALYZE records;
`);

await client.end();
console.log(`Done: ${grandTotal} rows total.`);
