"use client";

// Interactive chart builder: the explorer's filter panel + chart panel,
// embedded in the epidemiology page so visitors can compose their own
// graphs against the live database.

import { useCallback, useEffect, useState } from "react";
import ChartsPanel from "@/components/ChartsPanel";
import FilterPanel, { Meta } from "@/components/FilterPanel";
import { fmtInt } from "@/lib/format";
import { EMPTY_FILTERS, ExplorerFilters } from "@/lib/state";

const SEARCHABLE = [
  "category", "breed", "sex", "age_text", "diagnosis", "stains",
  "charge_type", "diagnosis_category", "specific_lesions",
];

export default function EpiBuilder() {
  const [filters, setFilters] = useState<ExplorerFilters>(EMPTY_FILTERS);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMeta(null);
    fetch(`/api/meta?dataset=${filters.dataset}`)
      .then(async (r) => {
        const j = await r.json();
        if (!r.ok) throw new Error(j.error ?? "Failed to load metadata");
        setMeta(j);
      })
      .catch((e) => setError(e.message));
  }, [filters.dataset]);

  const onFilters = useCallback((next: ExplorerFilters) => setFilters(next), []);

  if (error) {
    return (
      <p className="py-10 text-center text-sm" style={{ color: "var(--muted)" }}>
        The live chart builder needs the database connection: {error}
      </p>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-[300px_1fr]">
      <aside>
        <FilterPanel filters={filters} meta={meta} searchable={SEARCHABLE} onChange={onFilters} />
        {meta && (
          <p className="mt-2 text-xs" style={{ color: "var(--muted)" }}>
            Charting against {fmtInt(meta.total)} records in the{" "}
            {filters.dataset === "main" ? "pathology reports" : "cytology"} dataset.
          </p>
        )}
      </aside>
      <div className="min-w-0">
        <ChartsPanel filters={filters} />
      </div>
    </div>
  );
}
