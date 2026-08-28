"use client";

import MultiSelect, { Option } from "@/components/MultiSelect";
import { label } from "@/lib/format";
import { ExplorerFilters, activeFilterCount } from "@/lib/state";

export interface Meta {
  total: number;
  admin?: boolean;
  ageRange: { min: number | null; max: number | null };
  facets: Record<string, Option[]>;
}

const FACETS: { col: string; title: string }[] = [
  { col: "category", title: "Taxon" },
  { col: "breed", title: "Species / breed" },
  { col: "sex", title: "Sex" },
  { col: "diagnosis_category", title: "Body system" },
  { col: "specific_lesions", title: "Disease process" },
  { col: "charge_type", title: "Service type" },
];

export default function FilterPanel({
  filters,
  meta,
  searchable,
  onChange,
}: {
  filters: ExplorerFilters;
  meta: Meta | null;
  searchable: string[];
  onChange: (next: ExplorerFilters) => void;
}) {
  const set = (patch: Partial<ExplorerFilters>) =>
    onChange({ ...filters, ...patch });
  const setFacet = (col: string, values: string[]) =>
    set({ facets: { ...filters.facets, [col]: values } });

  const nActive = activeFilterCount(filters);

  return (
    <div className="card flex flex-col gap-4 p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Search &amp; filters</h2>
        {nActive > 0 && (
          <button
            type="button"
            className="text-xs font-medium"
            style={{ color: "var(--accent)" }}
            onClick={() =>
              set({ q: "", facets: {}, ageMin: "", ageMax: "" })
            }
          >
            Reset all ({nActive})
          </button>
        )}
      </div>

      <div>
        <div className="mb-1 text-xs font-medium" style={{ color: "var(--ink-2)" }}>
          Search terms
        </div>
        <input
          className="input"
          placeholder="e.g. carcinoma or lymphoma"
          value={filters.q}
          onChange={(e) => set({ q: e.target.value })}
        />
        <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          Join terms with <b>or</b> (or a comma) to match any, <b>and</b> to
          match all.
        </p>
      </div>

      <MultiSelect
        label="Columns to search in"
        options={searchable.map((c) => ({ value: c, name: label(c) }))}
        selected={filters.searchCols}
        onChange={(v) => set({ searchCols: v })}
        placeholder="Choose columns"
      />

      <div className="grid grid-cols-1 gap-3">
        {FACETS.map(({ col, title }) => (
          <MultiSelect
            key={col}
            label={title}
            options={meta?.facets[col] ?? []}
            selected={filters.facets[col] ?? []}
            onChange={(v) => setFacet(col, v)}
          />
        ))}
      </div>

      <div>
        <div className="mb-1 text-xs font-medium" style={{ color: "var(--ink-2)" }}>
          {label("age")}
          {meta?.ageRange.max != null && (
            <span style={{ color: "var(--muted)" }}>
              {" "}
              (data: {meta.ageRange.min}–{meta.ageRange.max})
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <input
            className="input"
            type="number"
            min={0}
            placeholder="min"
            value={filters.ageMin}
            onChange={(e) => set({ ageMin: e.target.value })}
          />
          <span style={{ color: "var(--muted)" }}>–</span>
          <input
            className="input"
            type="number"
            min={0}
            placeholder="max"
            value={filters.ageMax}
            onChange={(e) => set({ ageMax: e.target.value })}
          />
        </div>
        <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          Rows with unknown age are always kept.
        </p>
      </div>

      <div>
        <div className="mb-1 text-xs font-medium" style={{ color: "var(--ink-2)" }}>
          {label("tissues")}
        </div>
        <div className="flex items-center gap-2">
          <input
            className="input"
            type="number"
            min={0}
            placeholder="min"
            value={filters.tissuesMin}
            onChange={(e) => set({ tissuesMin: e.target.value })}
          />
          <span style={{ color: "var(--muted)" }}>–</span>
          <input
            className="input"
            type="number"
            min={0}
            placeholder="max"
            value={filters.tissuesMax}
            onChange={(e) => set({ tissuesMax: e.target.value })}
          />
        </div>
        <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          Number of organs submitted — low counts are typically biopsies, high
          counts necropsies.
        </p>
      </div>
    </div>
  );
}
