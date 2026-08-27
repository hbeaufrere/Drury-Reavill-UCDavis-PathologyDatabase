#!/usr/bin/env python3
"""Convert the committed parquet files into cleaned NDJSON.gz seed files.

Run once whenever the parquet data changes, then re-run `npm run db:seed`.

    python3 scripts/prepare_seed.py

Cleaning applied (light-touch, documented so it can be audited):
  - collapse repeated whitespace in category/breed ("Avian,  Psittacine")
  - strip Excel carriage-return artifacts (_x000D_) from diagnosis text
  - normalize sex values (trailing punctuation/digit typos, "Not Provid")
  - fold case-duplicate diagnosis_category values ("skin" -> "Skin")
  - null out physiologically impossible ages (> 200 years, data-entry noise)
"""
import gzip
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data")

SEX_MAP = {
    "female`": "Female", "female4": "Female", "female": "Female",
    "male`": "Male", "male.": "Male", "male": "Male",
    "neuter-f`": "Neuter-F", "neuter-f": "Neuter-F",
    "neuter-m": "Neuter-M",
    "not provid": "Not Provided", "not provided": "Not Provided",
    "unknown": "Unknown",
}

DIAG_CAT_MAP = {"skin": "Skin"}


def clean_text(v):
    if pd.isna(v):
        return None
    s = str(v).replace("_x000D_", "").replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s or None


def clean_sex(v):
    s = clean_text(v)
    if s is None:
        return None
    return SEX_MAP.get(s.lower(), s)


def clean_diag_cat(v):
    s = clean_text(v)
    if s is None:
        return None
    return DIAG_CAT_MAP.get(s, s)


def prepare(parquet_name, dataset_key):
    df = pd.read_parquet(os.path.join(ROOT, parquet_name))
    rows = []
    for r in df.itertuples(index=False):
        age = None if pd.isna(r.age) or r.age > 200 else round(float(r.age), 2)
        stains_charge = None if pd.isna(r.stains_charge) else float(r.stains_charge)
        rows.append({
            "dataset": dataset_key,
            "animal_name": clean_text(r.animal_name),
            "category": clean_text(r.category),
            "breed": clean_text(r.breed),
            "sex": clean_sex(r.sex),
            "age": age,
            "age_text": clean_text(r.age_text),
            "diagnosis": clean_text(r.diagnosis),
            "tissues": None if pd.isna(r.tissues) else int(r.tissues),
            "stains": clean_text(r.stains),
            "stains_charge": stains_charge,
            "charge_type": clean_text(r.charge_type),
            "diagnosis_category": clean_diag_cat(r.diagnosis_category),
            "specific_lesions": clean_text(r.specific_lesions),
        })
    out = os.path.join(OUT_DIR, f"{dataset_key}.ndjson.gz")
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{out}: {len(rows)} rows")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    prepare("reports_main.parquet", "main")
    # cyto_reports.parquet is a byte-for-byte duplicate of reports_main.parquet,
    # so only the main reports and the cytology file are loaded.
    prepare("cyto_cytology.parquet", "cytology")
