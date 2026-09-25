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


# ---------------------------------------------------------------------------
# Public-text anonymization. The full report text is kept in diagnosis_admin
# (admin sign-in only); the public `diagnosis` column has patient, client,
# veterinarian, and facility identifiers redacted. Heuristic by design:
# narrative reports are free text, so redaction rules are best-effort and
# the placeholders make any residual review easy to spot.
# ---------------------------------------------------------------------------

FACILITY_WORDS = (
    r"(?:Animal|Veterinary|Exotic|Pet|Bird|Avian|Emergency)?\s*"
    r"(?:Hospital|Clinic|Veterinary\s+Center|Medical\s+Center|Sanctuary|Rescue|"
    r"Aviary|Zoo|Aquarium)"
)

RE_ACCESSION = re.compile(r"\bV\d{5,7}\b(\s*VREM-?\d*)?", re.IGNORECASE)
RE_DR = re.compile(r"\b(?:Drs?\.?|Doctor)\s+[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)?")
RE_DVM = re.compile(r"\b[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?,?\s+(?:DVM|VMD)\b")
RE_FACILITY = re.compile(
    r"\b(?:[A-Z][A-Za-z'’&-]+\s+){1,4}" + FACILITY_WORDS + r"\b"
)
RE_OWNER = re.compile(
    r"\b(?:owned\s+by|owner[,:]?\s+(?:is\s+)?|client[,:]?\s+(?:is\s+)?)"
    r"(?:Mrs?\.\s+|Ms\.\s+)?[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)?",
    re.IGNORECASE,
)
RE_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")


def scrub_public(text, animal_name):
    """Redact identifying info from the public copy of the report text."""
    if text is None:
        return None
    s = text
    # The patient's own name, optionally followed by an owner surname
    # ("Leo Fitzgerald" -> "[patient]"). Short names (<3 chars) skipped.
    if animal_name and len(animal_name) >= 3:
        # Case-insensitive on the name itself; the optional trailing word is
        # only eaten when it is Capitalized (an owner surname).
        s = re.sub(
            r"\b(?i:" + re.escape(animal_name) + r")(\s+[A-Z][a-z'’-]+)?\b",
            "[patient]",
            s,
        )
    s = RE_ACCESSION.sub("[case no]", s)
    s = RE_DR.sub("[veterinarian]", s)
    s = RE_DVM.sub("[veterinarian]", s)
    s = RE_FACILITY.sub("[facility]", s)
    s = RE_OWNER.sub("[owner]", s)
    s = RE_PHONE.sub("[phone]", s)
    s = RE_EMAIL.sub("[email]", s)
    return s


def prepare(parquet_name, dataset_key):
    df = pd.read_parquet(os.path.join(ROOT, parquet_name))
    rows = []
    for r in df.itertuples(index=False):
        age = None if pd.isna(r.age) or r.age > 200 else round(float(r.age), 2)
        stains_charge = None if pd.isna(r.stains_charge) else float(r.stains_charge)
        animal_name = clean_text(r.animal_name)
        diagnosis_full = clean_text(r.diagnosis)
        rows.append({
            "dataset": dataset_key,
            "animal_name": animal_name,
            "category": clean_text(r.category),
            "breed": clean_text(r.breed),
            "sex": clean_sex(r.sex),
            "age": age,
            "age_text": clean_text(r.age_text),
            "diagnosis": scrub_public(diagnosis_full, animal_name),
            "diagnosis_admin": diagnosis_full,
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
