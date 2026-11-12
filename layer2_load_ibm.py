#!/usr/bin/env python3
"""
Layer 2, step 1: load the IBM HR Analytics attrition dataset into PostgreSQL.

NOTE: this dataset is SYNTHETIC (created by IBM data scientists). Use it to
build and demonstrate the modelling method, and say so in the README. The
real-data evidence is Layer 1.

Run from the project folder:
    $env:DATABASE_URL="postgresql+psycopg2://postgres:PASSWORD@localhost:5432/hr_attrition"
    python layer2_load_ibm.py
"""
import os
import re
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "raw" / "ibm_hr" / "ibm_hr_attrition.csv"
URL = ("https://raw.githubusercontent.com/pplonski/datasets-for-start/"
       "refs/heads/master/employee_attrition/HR-Employee-Attrition-All.csv")
CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]  # same value for every row


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def download() -> None:
    if CSV_PATH.exists():
        return
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    CSV_PATH.write_bytes(r.content)
    print(f"downloaded {CSV_PATH} ({len(r.content):,} bytes)")


def main() -> None:
    download()
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")  # file starts with a BOM
    df = df.drop(columns=CONSTANT_COLS)
    df.columns = [snake(c) for c in df.columns]
    df["attrition_flag"] = (df["attrition"] == "Yes").astype(int)

    # quick quality checks before loading
    assert df["employee_number"].is_unique, "duplicate employee ids"
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    print(f"{len(df)} employees, {df.shape[1]} columns, "
          f"attrition rate {df['attrition_flag'].mean():.1%}")

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("Set DATABASE_URL first (see the top of this file).")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
        if conn.execute(text("SELECT to_regclass('raw.ibm_hr_employee')")).scalar():
            conn.execute(text("TRUNCATE raw.ibm_hr_employee"))  # keep the table: views depend on it
    df.to_sql("ibm_hr_employee", engine, schema="raw", if_exists="append", index=False)
    print(f"loaded {len(df)} rows into raw.ibm_hr_employee")


if __name__ == "__main__":
    main()
