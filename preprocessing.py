#!/usr/bin/env python3
"""Preprocess district-quarter panel data for Moscow flat prices.

Script steps:
1) Read input CSV.
2) Clean district names from extra quotes/spaces.
3) Build a proper quarterly timestamp.
4) Validate uniqueness of district_name × quarter.
5) Compute district-specific price lags for 4, 8, ..., 32 quarters.
6) Compute abs_change, pct_change, CAGR for each window.
7) Add quality flags.
8) Save to Parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Windows in quarters: 1, 2, ..., 8 years.
LAG_WINDOWS = list(range(4, 33, 4))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess district-level quarterly price panel and export to Parquet."
    )
    parser.add_argument("input_csv", type=Path, help="Path to source CSV file")
    parser.add_argument("output_parquet", type=Path, help="Path to output Parquet file")
    parser.add_argument(
        "--min-deals",
        type=int,
        default=5,
        help="Minimum number of deals to mark an observation as reliable (default: 5)",
    )
    return parser.parse_args()


def clean_district_name(series: pd.Series) -> pd.Series:
    """Remove extra quotes and surrounding whitespace from district names."""
    cleaned = (
        series.astype("string")
        .str.replace('"', "", regex=False)
        .str.replace("'", "", regex=False)
        .str.strip()
    )
    return cleaned


def build_quarter_timestamp(df: pd.DataFrame) -> pd.Series:
    """Create quarter timestamp (quarter start date) from year and quarter number."""
    year = pd.to_numeric(df["year"], errors="coerce")
    q_num = pd.to_numeric(df["q_num"], errors="coerce")

    missing_year = year.isna()
    if missing_year.any():
        raise ValueError(
            f"Found {int(missing_year.sum())} rows with invalid year values; cannot build quarter timestamp."
        )

    missing_quarter = q_num.isna()
    if missing_quarter.any():
        raise ValueError(
            f"Found {int(missing_quarter.sum())} rows with invalid q_num values; expected 1..4."
        )

    invalid_q = ~q_num.isin([1, 2, 3, 4])
    if invalid_q.any():
        bad_vals = sorted(df.loc[invalid_q, "q_num"].dropna().unique().tolist())
        raise ValueError(f"Found invalid q_num values (expected 1..4): {bad_vals}")

    # PeriodIndex keeps quarter semantics; timestamp uses quarter start dates.
    quarter_period = pd.PeriodIndex(year=year.astype("Int64"), quarter=q_num.astype("Int64"), freq="Q")
    return quarter_period.to_timestamp(how="start")


def validate_unique_key(df: pd.DataFrame) -> None:
    """Ensure district_name × quarter is unique, otherwise raise descriptive error."""
    key_cols = ["district_name", "quarter"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        duplicates = (
            df.loc[dup_mask, key_cols]
            .value_counts()
            .rename("n_rows")
            .reset_index()
            .sort_values("n_rows", ascending=False)
        )
        sample = duplicates.head(10).to_string(index=False)
        raise ValueError(
            "Key district_name × quarter is not unique. "
            f"Found {len(duplicates)} duplicated key values. Top duplicates:\n{sample}"
        )


def add_lag_features(df: pd.DataFrame, min_deals: int) -> pd.DataFrame:
    """Add lagged prices, changes, CAGR, and quality flags for each window."""
    df = df.sort_values(["district_name", "quarter_ts"]).copy()

    # Ensure numeric columns are numeric for arithmetic operations.
    df["price_per_m2"] = pd.to_numeric(df["price_per_m2"], errors="coerce")
    df["n_deals"] = pd.to_numeric(df["n_deals"], errors="coerce")

    # Current-period quality flag.
    df["enough_deals_current"] = df["n_deals"] >= min_deals

    for lag in LAG_WINDOWS:
        years = lag // 4
        lag_price_col = f"price_lag_{lag}q"
        lag_deals_col = f"n_deals_lag_{lag}q"
        abs_col = f"abs_change_{lag}q"
        pct_col = f"pct_change_{lag}q"
        cagr_col = f"cagr_{lag}q"
        enough_base_col = f"enough_deals_base_{lag}q"
        valid_col = f"valid_growth_observation_{lag}q"

        base = df[["district_name", "quarter_ts", "price_per_m2", "n_deals"]].copy()
        base["quarter_ts"] = base["quarter_ts"] + pd.offsets.QuarterBegin(lag)
        base = base.rename(
            columns={
                "price_per_m2": lag_price_col,
                "n_deals": lag_deals_col,
            }
        )
        df = df.merge(base, on=["district_name", "quarter_ts"], how="left")

        # Absolute change in RUB per m².
        df[abs_col] = df["price_per_m2"] - df[lag_price_col]

        # Percent change relative to lag price.
        valid_base_price = df[lag_price_col] > 0
        df[pct_col] = np.where(
            valid_base_price,
            (df["price_per_m2"] / df[lag_price_col] - 1.0) * 100.0,
            np.nan,
        )

        # CAGR over N years: (P_t / P_{t-lag}) ** (1/N) - 1.
        growth_ratio = np.where(valid_base_price, df["price_per_m2"] / df[lag_price_col], np.nan)
        positive_ratio = growth_ratio > 0
        df[cagr_col] = np.where(
            positive_ratio,
            (growth_ratio ** (1 / years) - 1.0) * 100.0,
            np.nan,
        )

        # Quality flags for base observation and final valid growth record.
        df[enough_base_col] = df[lag_deals_col] >= min_deals
        df[valid_col] = (
            df["enough_deals_current"]
            & df[enough_base_col]
            & df["price_per_m2"].notna()
            & df[lag_price_col].notna()
            & (df["price_per_m2"] > 0)
            & (df[lag_price_col] > 0)
        )

    return df


def preprocess(input_csv: Path, output_parquet: Path, min_deals: int = 5) -> None:
    """Run the full preprocessing pipeline and write Parquet output."""
    df = pd.read_csv(input_csv)

    required_cols = {
        "district_name",
        "quarter",
        "year",
        "q_num",
        "n_deals",
        "price_per_m2",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df["district_name"] = clean_district_name(df["district_name"])
    df["quarter_ts"] = build_quarter_timestamp(df)

    validate_unique_key(df)

    result = add_lag_features(df=df, min_deals=min_deals)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_parquet, index=False)


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_csv=args.input_csv,
        output_parquet=args.output_parquet,
        min_deals=args.min_deals,
    )
