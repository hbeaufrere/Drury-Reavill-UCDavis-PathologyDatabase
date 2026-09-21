import Link from "next/link";
import epi from "@/data/epi.json";
import EpiBuilder from "@/components/EpiBuilder";
import {
  AssociationChart,
  CLS_COLOR,
  HBarList,
  StackedShare,
  SystemHeatmap,
  TumorAgeLines,
} from "@/components/EpiCharts";
import { fmtInt } from "@/lib/format";

export const metadata = {
  title: "Epidemiology of Diseases in Exotic Animals — Drury R. Reavill Pathology Database",
  description:
    "Disease trends across 84,000+ exotic animal pathology records: neoplasia, infections, age patterns, and species–disease associations.",
};

function Section({
  title,
  intro,
  children,
}: {
  title: string;
  intro: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card p-6 lg:p-8">
      <h2 className="text-base font-bold">{title}</h2>
      <p className="mb-6 mt-1 max-w-3xl text-sm" style={{ color: "var(--ink-2)" }}>
        {intro}
      </p>
      {children}
    </section>
  );
}

export default function EpidemiologyPage() {
  const t = epi.totals;

  return (
    <div>
      <header style={{ background: "var(--ucd-blue)" }}>
        <div className="mx-auto max-w-6xl px-4 py-12 lg:px-8">
          <p className="mb-2 text-xs font-bold uppercase tracking-widest" style={{ color: "var(--ucd-gold)" }}>
            <Link href="/">Drury R. Reavill Pathology Database</Link>
          </p>
          <h1 className="max-w-3xl text-3xl font-bold tracking-tight text-white lg:text-4xl">
            Epidemiology of Diseases in Exotic Animals
          </h1>
          <p className="mt-3 max-w-2xl text-base" style={{ color: "#c9d4e4" }}>
            Patterns mined from {fmtInt(t.records)} pathology records —{" "}
            {fmtInt(t.classified_process)} with a classified disease process and{" "}
            {fmtInt(t.with_age)} with a usable age. Counts based on diagnosis-text
            keywords are estimates, not audited case series.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href="/explore"
              className="inline-flex items-center gap-2 rounded-lg px-5 py-2.5 text-sm font-bold"
              style={{ background: "var(--ucd-gold)", color: "var(--ucd-blue)" }}
            >
              Search the database →
            </Link>
            <a
              href="#builder"
              className="inline-flex items-center gap-2 rounded-lg border px-5 py-2.5 text-sm font-bold text-white"
              style={{ borderColor: "rgba(255,255,255,0.4)" }}
            >
              Build your own charts ↓
            </a>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-4 py-10 lg:px-8">
        <Section
          title="What brings each class to the pathologist"
          intro="Share of classified diagnoses by disease process. Tumors dominate the mammal caseload (41.5% of classified diagnoses) but are far less frequent in reptiles and amphibians, where infectious and inflammatory disease lead."
        >
          <StackedShare
            rows={epi.processMix}
            keys={["Tumor", "Inflammatory", "Infection", "Metabolic/degenerative", "Other"]}
            extraKey={null}
          />
        </Section>

        <Section
          title="Tumor risk climbs steeply with age"
          intro="Among diagnoses with a classified process and a usable age, the tumor share rises from under 4% in animals below one year to roughly half of diagnoses in older mammals. Points are omitted where fewer than 40 records fall in a bin."
        >
          <TumorAgeLines rows={epi.tumorByAge} />
        </Section>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <Section
            title="Species with the highest tumor burden"
            intro="Tumor share of classified diagnoses in species with at least 250 submissions. Ferrets top the archive — nearly 6 of every 10 classified diagnoses are neoplastic."
          >
            <HBarList
              rows={epi.tumorBySpecies.map((r) => ({
                label: r.species,
                value: r.pct,
                note: `n=${fmtInt(r.n)}`,
                color: CLS_COLOR[r.cls],
              }))}
              suffix="%"
            />
          </Section>

          <Section
            title="Most common named tumor types"
            intro="Archive-wide counts of tumor entities named in the diagnosis text. Lymphoma leads across classes; squamous cell carcinoma and the reptile pigment-cell (chromatophoroma) family follow."
          >
            <HBarList
              rows={epi.tumorEntities.map((r) => ({
                label: r.entity,
                value: r.n,
                color: "var(--cls-1)",
              }))}
            />
          </Section>
        </div>

        <Section
          title="Signature species–disease associations"
          intro="Conditions strongly over-represented in one species relative to the rest of its class (bar length on a log scale; minimum 10 cases). The screen recovers classic associations — budgerigar pituitary tumors, boa inclusion body disease — alongside less-reported signals such as leopard gecko xanthomatosis."
        >
          <AssociationChart rows={epi.associations} />
        </Section>

        <Section
          title="Which body systems are affected, class by class"
          intro="Percentage of classified diagnoses assigned to each organ system. Skin disease dominates bird and reptile submissions; digestive and systemic disease weigh heavier in mammals and fish."
        >
          <SystemHeatmap systems={epi.bodySystems.systems} rows={epi.bodySystems.rows} />
        </Section>

        <Section
          title="Infections: which agents, in which classes"
          intro="Among diagnoses classified as infectious, the causative agent group recorded by the pathologist. Fungal disease looms large in reptiles; viral infections take a bigger share in birds."
        >
          <StackedShare
            rows={epi.infectionAgents}
            keys={["Bacteria", "Fungus", "Virus", "Protozoa", "Metazoan", "Yeast"]}
            extraKey="Unspecified"
          />
        </Section>

        <section id="builder" className="card p-6 lg:p-8">
          <h2 className="text-base font-bold">Build your own charts</h2>
          <p className="mb-6 mt-1 max-w-3xl text-sm" style={{ color: "var(--ink-2)" }}>
            Compose bar, donut, histogram, box, scatter, or heatmap views against
            the live database. Add search terms and filters on the left — every
            chart recomputes from the records that match. Switch to the{" "}
            <Link href="/explore" style={{ color: "var(--accent)" }}>
              full explorer
            </Link>{" "}
            to see and download the underlying cases.
          </p>
          <EpiBuilder />
        </section>

        <footer
          className="border-t pt-5 text-center text-xs"
          style={{ borderColor: "var(--grid)", color: "var(--muted)" }}
        >
          Keyword-derived counts are screening estimates and require case-level
          review before formal use ·{" "}
          <Link href="/#cite" style={{ color: "var(--accent)" }}>
            How to cite
          </Link>{" "}
          ·{" "}
          <Link href="/" style={{ color: "var(--accent)" }}>
            Back to overview
          </Link>
        </footer>
      </div>
    </div>
  );
}
