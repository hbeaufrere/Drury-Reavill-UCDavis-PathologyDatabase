"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { fmtInt } from "@/lib/format";

export interface Option {
  value: string;
  name?: string;
  count?: number;
}

export default function MultiSelect({
  label,
  options,
  selected,
  onChange,
  placeholder = "All",
}: {
  label: string;
  options: Option[];
  selected: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase();
    const list = f
      ? options.filter((o) => (o.name ?? o.value).toLowerCase().includes(f))
      : options;
    return list.slice(0, 400);
  }, [options, filter]);

  const nameOf = (v: string) => options.find((o) => o.value === v)?.name ?? v;

  const toggle = (v: string) => {
    onChange(
      selected.includes(v) ? selected.filter((x) => x !== v) : [...selected, v]
    );
  };

  return (
    <div ref={ref} className="relative">
      <div className="mb-1 text-xs font-medium" style={{ color: "var(--ink-2)" }}>
        {label}
      </div>
      <button
        type="button"
        className="input flex items-center justify-between gap-2 text-left"
        onClick={() => setOpen((o) => !o)}
      >
        <span
          className="truncate"
          style={{ color: selected.length ? "var(--ink)" : "var(--muted)" }}
        >
          {selected.length === 0
            ? placeholder
            : selected.length <= 2
              ? selected.map(nameOf).join(", ")
              : `${selected.length} selected`}
        </span>
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
          <path
            d="M2 4l4 4 4-4"
            fill="none"
            stroke="var(--muted)"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      </button>
      {open && (
        <div
          className="card absolute z-20 mt-1 w-full min-w-56 p-2 shadow-lg"
          style={{ maxHeight: "20rem", display: "flex", flexDirection: "column" }}
        >
          {options.length > 12 && (
            <input
              className="input mb-2"
              placeholder="Type to filter…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              autoFocus
            />
          )}
          {selected.length > 0 && (
            <button
              type="button"
              className="mb-1 self-start text-xs font-medium"
              style={{ color: "var(--accent)" }}
              onClick={() => onChange([])}
            >
              Clear selection
            </button>
          )}
          <div className="overflow-y-auto">
            {shown.map((o) => (
              <label
                key={o.value}
                className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm hover:bg-[color-mix(in_srgb,var(--accent)_8%,transparent)]"
              >
                <input
                  type="checkbox"
                  checked={selected.includes(o.value)}
                  onChange={() => toggle(o.value)}
                />
                <span className="flex-1 truncate">{o.name ?? o.value}</span>
                {o.count !== undefined && (
                  <span className="tabular text-xs" style={{ color: "var(--muted)" }}>
                    {fmtInt(o.count)}
                  </span>
                )}
              </label>
            ))}
            {shown.length === 0 && (
              <div className="px-2 py-1 text-sm" style={{ color: "var(--muted)" }}>
                No matches
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
