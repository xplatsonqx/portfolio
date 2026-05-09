from __future__ import annotations

from dataclasses import dataclass
import warnings

import pandas as pd


ROLE_NUMERICAL = "numerical"
ROLE_CATEGORICAL = "categorical"
ROLE_DATETIME = "datetime"
ROLE_IDENTIFIER = "identifier"
ROLE_TEXT = "text"
ROLE_BOOLEAN = "boolean"


_ID_HINTS = (
    "id",
    "patient",
    "invoice",
    "protocol",
    "number",
    "no",
    "code",
    "uuid",
    "guid",
)

_DATETIME_HINTS = ("date", "time", "datetime", "timestamp")


def _looks_boolean(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().unique()
        if len(vals) <= 2 and set(vals).issubset({0, 1}):
            return True
    return False


def _try_parse_datetime(s: pd.Series) -> tuple[bool, pd.Series | None]:
    # Sample to keep it fast on large datasets
    sample = s.dropna().astype(str).head(2000)
    if sample.empty:
        return False, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    ok_ratio = float(parsed.notna().mean())
    if ok_ratio >= 0.9:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            full = pd.to_datetime(s, errors="coerce")
        return True, full
    return False, None


def _string_avg_len(s: pd.Series) -> float:
    sample = s.dropna().astype(str).head(2000)
    if sample.empty:
        return 0.0
    return float(sample.map(len).mean())


def infer_role(s: pd.Series, col_name: str, n_rows: int) -> tuple[str, dict]:
    name = (col_name or "").strip().lower()
    hints = {
        "id_hint": any(h in name for h in _ID_HINTS),
        "datetime_hint": any(h in name for h in _DATETIME_HINTS),
    }

    if _looks_boolean(s):
        return ROLE_BOOLEAN, hints

    if pd.api.types.is_datetime64_any_dtype(s):
        return ROLE_DATETIME, hints

    # Parse only when there is a datetime-like name hint to avoid noisy false checks.
    is_dt, _ = _try_parse_datetime(s) if hints["datetime_hint"] else (False, None)
    if is_dt:
        return ROLE_DATETIME, hints

    missing = int(s.isna().sum())
    non_null = n_rows - missing
    unique = int(s.nunique(dropna=True))
    unique_ratio = (unique / non_null) if non_null else 0.0
    hints["unique_ratio_non_null"] = unique_ratio

    if pd.api.types.is_numeric_dtype(s):
        # Heuristic: high-uniqueness integer-like columns are often identifiers
        is_int_like = pd.api.types.is_integer_dtype(s) or (s.dropna().astype(float) % 1 == 0).all()
        hints["is_int_like"] = bool(is_int_like)
        if unique_ratio >= 0.95 and (hints["id_hint"] or is_int_like):
            return ROLE_IDENTIFIER, hints
        return ROLE_NUMERICAL, hints

    # object / string-like
    avg_len = _string_avg_len(s)
    hints["avg_len"] = avg_len
    if unique_ratio >= 0.95 or hints["id_hint"]:
        return ROLE_IDENTIFIER, hints
    if avg_len >= 30 or unique_ratio >= 0.8:
        return ROLE_TEXT, hints
    return ROLE_CATEGORICAL, hints


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        missing_count = int(s.isna().sum())
        missing_pct = (missing_count / n_rows * 100.0) if n_rows else 0.0
        n_unique = int(s.nunique(dropna=True))
        role, hints = infer_role(s, str(col), n_rows)
        example_vals = s.dropna().head(3).tolist()
        example_vals = [str(v) for v in example_vals]

        # High-cardinality warning for ID-like / text-like / extreme categorical
        high_cardinality = False
        if n_rows:
            unique_ratio_rows = n_unique / n_rows
            if role in {ROLE_IDENTIFIER, ROLE_TEXT} and unique_ratio_rows >= 0.5:
                high_cardinality = True
            if role == ROLE_CATEGORICAL and unique_ratio_rows >= 0.2:
                high_cardinality = True

        rows.append(
            {
                "column": str(col),
                "dtype": dtype,
                "role": role,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "n_unique": n_unique,
                "example_values": ", ".join(example_vals),
                "high_cardinality": high_cardinality,
            }
        )
    return pd.DataFrame(rows)

