#!/usr/bin/env python3
"""Precompute the epidemiology-report dataset (data/epi.json).

The archive is static, so the /epidemiology report is generated offline:

    python3 scripts/build_epi.py    # needs: pip install pandas pyarrow

Re-run whenever the parquet data changes, then commit data/epi.json.
"""
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Category -> taxonomic class (mirror of lib/taxonomy.ts, condensed).
def to_class(cat):
    if pd.isna(cat):
        return None
    c = str(cat)
    if c.startswith("Avian"):
        return "Birds"
    if c.startswith("Reptile"):
        return "Reptiles"
    if c.startswith("Amphibian"):
        return "Amphibians"
    if c.startswith("Fish"):
        return "Fish"
    if c in ("Cnidaria Phylum", "Bugs", "Echinoderm"):
        return "Invertebrates"
    if c in ("Not Provided", "Not provided", "Misc Category"):
        return None
    return "Mammals"


def process_group(lesion):
    if pd.isna(lesion):
        return None
    s = str(lesion)
    if s.startswith("Infection"):
        return "Infection"
    if s.startswith("Metabolic") or s.startswith("Degenerative"):
        return "Metabolic/degenerative"
    if s in ("Tumor",):
        return "Tumor"
    if s in ("Inflammatory",):
        return "Inflammatory"
    return "Other"


def infection_agent(lesion):
    s = str(lesion).lower()
    m = re.search(r"infection \((\w+)", s)
    if not m:
        return "Unspecified"
    a = m.group(1)
    return {
        "bacteria": "Bacteria", "baceria": "Bacteria", "virus": "Virus",
        "fungus": "Fungus", "yeast": "Yeast", "protozoa": "Protozoa",
        "metazoan": "Metazoan", "ibd": "Virus",
    }.get(a, "Other")


df = pd.read_parquet(os.path.join(ROOT, "reports_main.parquet"))
df["cls"] = df["category"].map(to_class)
df["proc"] = df["specific_lesions"].map(process_group)
df["diag"] = df["diagnosis"].fillna("").str.lower()
df["age_c"] = df["age"].where((df["age"] > 0) & (df["age"] <= 30))

CLASSES = ["Birds", "Mammals", "Reptiles", "Fish", "Amphibians"]
out = {}

# 1. Disease-process mix per class (rows with a classified process).
PROCS = ["Tumor", "Inflammatory", "Infection", "Metabolic/degenerative", "Other"]
mix = []
for cl in CLASSES:
    sub = df[(df["cls"] == cl) & df["proc"].notna()]
    if len(sub) < 100:
        continue
    row = {"cls": cl, "n": len(sub)}
    for p in PROCS:
        row[p] = round(100 * (sub["proc"] == p).mean(), 1)
    mix.append(row)
out["processMix"] = mix

# 2. Infection agents per class.
agents = []
AG = ["Bacteria", "Fungus", "Virus", "Protozoa", "Metazoan", "Yeast"]
for cl in CLASSES:
    sub = df[(df["cls"] == cl) & df["specific_lesions"].fillna("").str.startswith("Infection")]
    if len(sub) < 50:
        continue
    ag = sub["specific_lesions"].map(infection_agent)
    row = {"cls": cl, "n": len(sub)}
    for a in AG:
        row[a] = round(100 * (ag == a).mean(), 1)
    agents.append(row)
out["infectionAgents"] = agents

# 3. Tumor share by age bin (classified-process rows with usable age).
bins = [(0, 1), (1, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 30)]
age_rows = []
base = df[df["proc"].notna() & df["age_c"].notna()]
series = {"All classes": base}
for cl in ["Birds", "Mammals", "Reptiles"]:
    series[cl] = base[base["cls"] == cl]
for lo, hi in bins:
    row = {"bin": f"{lo}–{hi}"}
    for name, sub in series.items():
        b = sub[(sub["age_c"] >= lo) & (sub["age_c"] < hi)]
        row[name] = round(100 * (b["proc"] == "Tumor").mean(), 1) if len(b) >= 40 else None
        row[f"{name}_n"] = len(b)
    age_rows.append(row)
out["tumorByAge"] = age_rows

# 4. Tumor proportional morbidity by species (n >= 250).
tum = []
AMBIGUOUS = {"Domestic", "Unknown", "Mix", "Mixed", "Not Provided", "Not provided"}
for sp, n in df["breed"].value_counts().items():
    if n < 250 or sp in AMBIGUOUS:
        continue
    sub = df[(df["breed"] == sp) & df["proc"].notna()]
    if len(sub) < 150:
        continue
    pct = 100 * (sub["proc"] == "Tumor").mean()
    tum.append({"species": sp, "cls": df.loc[df["breed"] == sp, "cls"].mode().iat[0],
                "n": len(sub), "pct": round(pct, 1)})
tum = sorted(tum, key=lambda r: -r["pct"])[:14]
out["tumorBySpecies"] = tum

# 5. Most common named tumor entities archive-wide.
ENTITIES = {
    "Lymphoma / leukemia": r"lymphoma|lymphosarcoma|leukemia",
    "Squamous cell carcinoma": r"squamous cell carcinoma",
    "Fibrosarcoma": r"fibrosarcoma",
    "Adenocarcinoma": r"adenocarcinoma",
    "Papilloma": r"papilloma",
    "Lipoma / liposarcoma": r"\blipoma|liposarcoma",
    "Chromatophoroma family": r"chromatophoroma|iridophoroma|melanophoroma|xanthophoroma",
    "Sertoli / seminoma": r"sertoli|seminoma",
    "Melanoma": r"melanoma\b",
    "Hemangioma / hemangiosarcoma": r"hemangio",
    "Myxoma / myxosarcoma": r"myxo(ma|sarcoma)",
    "Osteoma / osteosarcoma": r"osteosarcoma|osteoma\b",
    "Basal / trichoblastoma": r"basal cell|trichoblastoma",
    "Insulinoma": r"insulinoma",
    "Adrenal neoplasia": r"adrenal.{0,30}(adenoma|carcinoma|neoplas)",
}
ent = []
for name, pat in ENTITIES.items():
    hits = df["diag"].str.contains(pat, regex=True)
    ent.append({"entity": name, "n": int(hits.sum())})
out["tumorEntities"] = sorted(ent, key=lambda r: -r["n"])

# 6. Species–disease association screen (within-class baseline).
CONDS = {
    "Pituitary tumor": r"pituitary",
    "Renal tumor": r"renal (carcinoma|adenocarcinoma|adenoma)|nephroblastoma",
    "Testicular tumor": r"seminoma|sertoli|testicular (tumor|neoplas)",
    "Macrorhabdus (gastric yeast)": r"macrorhabdus|megabacteri",
    "Squamous cell carcinoma": r"squamous cell carcinoma",
    "Bornavirus / PDD": r"bornavir|ganglioneuritis|proventricular dilat",
    "Atherosclerosis": r"atheroscler",
    "Bile duct carcinoma": r"bile duct carcinoma|cholangiocarcinoma",
    "Cloacal papillomatosis": r"papillom",
    "Xanthomatosis": r"xanthoma",
    "Inclusion body disease": r"inclusion body disease|\bibd\b",
    "Cryptosporidiosis": r"cryptosporid",
    "Chromatophoroma": r"chromatophoroma|iridophoroma|melanophoroma",
    "Nannizziopsis (CANV)": r"nannizziopsis|\bcanv\b|yellow fungus",
    "Adenovirus": r"adenovir",
    "Uterine adenocarcinoma": r"uterine adenocarcinoma|endometrial (carcinoma|adenocarcinoma)",
    "Adrenal disease/neoplasia": r"adrenal",
    "Insulinoma": r"insulinoma",
    "Lymphoma": r"lymphoma",
    "Pancreatic disease": r"pancreat",
    "Myocardial disease": r"cardiomyopathy|myocard",
    "Gout": r"\bgout",
    "Spinal osteomyelitis": r"osteomyelitis|spinal osteopathy|vertebral osteo",
    "Hepatic lipidosis": r"hepatic lipidosis",
}
assoc = []
for cl in CLASSES:
    cdf = df[df["cls"] == cl]
    if len(cdf) < 500:
        continue
    counts = cdf["breed"].value_counts()
    species = [s for s in counts.index if counts[s] >= (80 if cl == "Reptiles" else 250)]
    for cond, pat in CONDS.items():
        hit = cdf["diag"].str.contains(pat, regex=True)
        tot = int(hit.sum())
        if tot < 15:
            continue
        for sp in species:
            m = cdf["breed"] == sp
            k = int((hit & m).sum())
            n_sp = int(m.sum())
            if k < 10:
                continue
            p_sp = k / n_sp
            p_rest = (tot - k) / (len(cdf) - n_sp)
            if p_rest <= 0:
                continue
            ratio = p_sp / p_rest
            if ratio >= 3.0:
                assoc.append({
                    "cls": cl, "species": sp, "condition": cond, "cases": k,
                    "pct": round(100 * p_sp, 1), "ratio": round(ratio, 1),
                })
assoc = sorted(assoc, key=lambda r: -r["ratio"])[:18]
out["associations"] = assoc

# 7. Body system share per class.
systems = []
SYS = ["Skin", "Digestive", "Systemic", "Respiratory", "Urinary", "Reproductive",
       "MusculoSkeletal", "CardioVascular", "HemeLymph", "Endocrine"]
for cl in CLASSES:
    sub = df[(df["cls"] == cl) & df["diagnosis_category"].notna()]
    if len(sub) < 100:
        continue
    row = {"cls": cl, "n": len(sub)}
    for s in SYS:
        row[s] = round(100 * (sub["diagnosis_category"] == s).mean(), 1)
    systems.append(row)
out["bodySystems"] = {"systems": SYS, "rows": systems}

out["totals"] = {
    "records": int(len(df)),
    "classified_process": int(df["proc"].notna().sum()),
    "with_age": int(df["age_c"].notna().sum()),
}

with open(os.path.join(ROOT, "data", "epi.json"), "w") as f:
    json.dump(out, f, indent=1)
print("data/epi.json written")
for k, v in out.items():
    print(f"  {k}: {len(v) if isinstance(v, list) else v if k=='totals' else '...'}")
