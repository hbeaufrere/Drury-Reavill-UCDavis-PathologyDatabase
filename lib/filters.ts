// Shared filter parsing and SQL fragment building for all API routes.
// Every column name that reaches SQL is validated against these whitelists;
// values only ever travel as bind parameters.

export const DATASETS = ["main", "cytology"] as const;
export type Dataset = (typeof DATASETS)[number];

export const TEXT_COLUMNS = [
  "animal_name",
  "category",
  "breed",
  "sex",
  "age_text",
  "diagnosis",
  "stains",
  "charge_type",
  "diagnosis_category",
  "specific_lesions",
] as const;

export const NUMERIC_COLUMNS = ["age", "tissues", "stains_charge"] as const;

export const ALL_COLUMNS = [
  "animal_name", "category", "breed", "sex", "age", "age_text", "diagnosis",
  "tissues", "stains", "stains_charge", "charge_type", "diagnosis_category",
  "specific_lesions",
] as const;

export const FACET_COLUMNS = [
  "category", "sex", "breed", "diagnosis_category", "specific_lesions",
  "charge_type",
] as const;

export const DEFAULT_SEARCH_COLUMNS = [
  "diagnosis", "category", "breed", "specific_lesions",
];

// Patient names stay in the database but are only visible to the admin.
const RESTRICTED_COLUMNS = ["animal_name"];

export function visibleColumns(admin: boolean): string[] {
  return admin
    ? [...ALL_COLUMNS]
    : ALL_COLUMNS.filter((c) => !RESTRICTED_COLUMNS.includes(c));
}

export function searchableColumns(admin: boolean): string[] {
  return admin
    ? [...TEXT_COLUMNS]
    : TEXT_COLUMNS.filter((c) => !RESTRICTED_COLUMNS.includes(c));
}

// Mirrors the original Streamlit behavior: "and" between all terms wins,
// otherwise "or"; a comma list means "or"; a plain string is one term.
export function parseSearchQuery(raw: string): { mode: "and" | "or"; terms: string[] } {
  const s = raw.trim();
  if (!s) return { mode: "or", terms: [] };
  const hasAnd = /\band\b/i.test(s);
  const hasOr = /\bor\b/i.test(s);
  const clean = (parts: string[]) => parts.map((p) => p.trim()).filter(Boolean);
  if (hasAnd) return { mode: "and", terms: clean(s.split(/\s+and\s+/i)) };
  if (hasOr) return { mode: "or", terms: clean(s.split(/\s+or\s+/i)) };
  if (s.includes(",")) return { mode: "or", terms: clean(s.split(",")) };
  return { mode: "or", terms: [s] };
}

function escapeLike(term: string): string {
  return term.replace(/[\\%_]/g, (c) => `\\${c}`);
}

export interface FilterSpec {
  dataset: Dataset;
  q: string;
  searchCols: string[];
  facets: Partial<Record<(typeof FACET_COLUMNS)[number], string[]>>;
  ageMin: number | null;
  ageMax: number | null;
  tissuesMin: number | null;
  tissuesMax: number | null;
}

// True when the request narrows the dataset at all — the public CSV export
// requires an actual search, never a full-table dump.
export function hasActiveFilters(f: FilterSpec): boolean {
  return (
    f.q.trim().length > 0 ||
    Object.values(f.facets).some((v) => v && v.length > 0) ||
    f.ageMin !== null ||
    f.ageMax !== null ||
    f.tissuesMin !== null ||
    f.tissuesMax !== null
  );
}

export function parseFilters(params: URLSearchParams, admin = false): FilterSpec {
  const ds = params.get("dataset");
  const dataset: Dataset = DATASETS.includes(ds as Dataset) ? (ds as Dataset) : "main";

  const allowed = searchableColumns(admin);
  const colsParam = params.get("cols");
  const requested = colsParam ? colsParam.split(",") : DEFAULT_SEARCH_COLUMNS;
  const searchCols = requested.filter((c) => allowed.includes(c));

  const facets: FilterSpec["facets"] = {};
  for (const col of FACET_COLUMNS) {
    const v = params.getAll(col).flatMap((x) => x.split("||")).filter(Boolean);
    if (v.length) facets[col] = v;
  }

  const num = (name: string) => {
    const v = params.get(name);
    if (v === null || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };

  return {
    dataset,
    q: params.get("q") ?? "",
    searchCols: searchCols.length ? searchCols : DEFAULT_SEARCH_COLUMNS,
    facets,
    ageMin: num("age_min"),
    ageMax: num("age_max"),
    tissuesMin: num("tissues_min"),
    tissuesMax: num("tissues_max"),
  };
}

export interface WhereClause {
  sql: string;
  params: unknown[];
}

export function buildWhere(f: FilterSpec): WhereClause {
  const conds: string[] = [];
  const params: unknown[] = [];
  const bind = (v: unknown) => {
    params.push(v);
    return `$${params.length}`;
  };

  conds.push(`dataset = ${bind(f.dataset)}`);

  const { mode, terms } = parseSearchQuery(f.q);
  if (terms.length && f.searchCols.length) {
    const termConds = terms.map((t) => {
      const p = bind(`%${escapeLike(t)}%`);
      const perCol = f.searchCols.map((c) => `${c} ILIKE ${p}`);
      return `(${perCol.join(" OR ")})`;
    });
    conds.push(`(${termConds.join(mode === "and" ? " AND " : " OR ")})`);
  }

  for (const [col, values] of Object.entries(f.facets)) {
    if (values && values.length) {
      conds.push(`${col} = ANY(${bind(values)})`);
    }
  }

  // Rows with unknown age are kept, matching the original app's behavior.
  if (f.ageMin !== null || f.ageMax !== null) {
    const lo = f.ageMin ?? 0;
    const hi = f.ageMax ?? 1000;
    conds.push(`(age IS NULL OR age BETWEEN ${bind(lo)} AND ${bind(hi)})`);
  }

  // Organ/tissue count filter (used to separate biopsies from necropsies).
  if (f.tissuesMin !== null || f.tissuesMax !== null) {
    const lo = f.tissuesMin ?? 0;
    const hi = f.tissuesMax ?? 10000;
    conds.push(`tissues BETWEEN ${bind(lo)} AND ${bind(hi)}`);
  }

  return { sql: conds.join(" AND "), params };
}
