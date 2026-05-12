from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_csv

DEFAULT_DROP_PATIENT_PREFIX = "P"


def convert_drug_sales_dataframe(
    df: pd.DataFrame,
    *,
    drop_drug_ser_no: bool = True,
    fill_recipe_mode: bool = True,
    fill_erecete_s: bool = True,
    drop_field_of_medicine: bool = True,
    drop_missing_drug_name: bool = True,
    drop_column_name: str | None = None,
    drop_patient_name_prefix: str | None = DEFAULT_DROP_PATIENT_PREFIX,
) -> pd.DataFrame:
    df = df.copy()

    if drop_drug_ser_no and "DrugSerNo" in df.columns:
        df = df.drop(columns=["DrugSerNo"])

    if fill_recipe_mode and "Recipe" in df.columns:
        recipe = df["Recipe"].astype("string").str.strip()
        mode = recipe.dropna().mode()
        fill_value = mode.iloc[0] if not mode.empty else "Itself"
        df["Recipe"] = recipe.fillna(fill_value)

    if fill_erecete_s and "ERecete" in df.columns:
        e = df["ERecete"].astype("string").str.strip()
        df["ERecete"] = e.fillna("S")

    if drop_field_of_medicine and "Field of Medicine" in df.columns:
        df = df.drop(columns=["Field of Medicine"])

    if drop_missing_drug_name and "Drug Name" in df.columns:
        df = df[df["Drug Name"].notna()].copy()

    if drop_column_name:
        c = str(drop_column_name).strip()
        if c and c in df.columns:
            df = df.drop(columns=[c])

    if drop_patient_name_prefix is not None and "Patient name" in df.columns:
        pfx = str(drop_patient_name_prefix).strip().upper()
        s = df["Patient name"].astype("string").str.strip()
        mask = s.str.upper().str.startswith(pfx, na=False)
        df = df[~mask].copy()

    return df


def convert_drug_sales_dataset(
    *,
    input_path: Path,
    output_path: Path,
    sep: str = ";",
) -> None:
    df = load_csv(input_path, sep=sep)
    df = convert_drug_sales_dataframe(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=sep, index=False)
