export const COLUMN_LABELS: Record<string, string> = {
  animal_name: "Patient name",
  category: "Taxon",
  breed: "Species / breed",
  sex: "Sex",
  age: "Age (years)",
  age_text: "Age note",
  diagnosis: "Diagnosis (full text)",
  tissues: "Organs (n)",
  stains: "Special stains",
  stains_charge: "Stain charge",
  charge_type: "Service type",
  diagnosis_category: "Body system",
  specific_lesions: "Disease process",
};

export function label(col: string): string {
  return COLUMN_LABELS[col] ?? col;
}

export function fmtInt(n: number): string {
  return n.toLocaleString("en-US");
}

export function fmtNum(n: number): string {
  if (Number.isInteger(n)) return fmtInt(n);
  return n.toLocaleString("en-US", { maximumFractionDigits: 2 });
}
