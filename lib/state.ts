import { DEFAULT_SEARCH_COLUMNS } from "@/lib/filters";

export interface ExplorerFilters {
  dataset: "main" | "cytology";
  q: string;
  searchCols: string[];
  facets: Record<string, string[]>;
  ageMin: string;
  ageMax: string;
}

export const EMPTY_FILTERS: ExplorerFilters = {
  dataset: "main",
  q: "",
  searchCols: DEFAULT_SEARCH_COLUMNS,
  facets: {},
  ageMin: "",
  ageMax: "",
};

// Serializes the filter state into the query string shared by the
// records, chart, and export endpoints.
export function filterParams(f: ExplorerFilters): URLSearchParams {
  const p = new URLSearchParams();
  p.set("dataset", f.dataset);
  if (f.q.trim()) p.set("q", f.q.trim());
  if (f.searchCols.length) p.set("cols", f.searchCols.join(","));
  for (const [col, values] of Object.entries(f.facets)) {
    if (values.length) p.set(col, values.join("||"));
  }
  if (f.ageMin !== "") p.set("age_min", f.ageMin);
  if (f.ageMax !== "") p.set("age_max", f.ageMax);
  return p;
}

export function activeFilterCount(f: ExplorerFilters): number {
  let n = 0;
  if (f.q.trim()) n++;
  for (const v of Object.values(f.facets)) if (v.length) n++;
  if (f.ageMin !== "" || f.ageMax !== "") n++;
  return n;
}
