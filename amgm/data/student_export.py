# This tool exports data for student projects, anonymizing the securities.

from pathlib import Path
from amgm.data.base import BaseAMData
from amgm import config as amgm_config
import polars as pl


def export_student_data(dset_path: Path, output_path: Path):

    D = BaseAMData(dset_path=dset_path, with_sector_codes=False)  # No date or iid filtering

    columns = {
        "IssueId": "Security",
        "Date": "Date",
        "TradingDay": "Day",
        "OpAdjUsd": "Open",
        "LoAdjUsd": "Low",
        "HiAdjUsd": "High",
        "ClAdjUsd": "Close",
        "VoAdj": "Volume",
    }

    # Extract and rename columns
    secs = D.securities.select([pl.col(col).alias(new_col) for col, new_col in columns.items()])
    # Convert some columns to appropriate types
    secs = secs.with_columns(pl.col("Date").dt.date().alias("Date"))  # Drop time
    secs = secs.with_columns(pl.col("Volume").cast(pl.Int64))  # Volume as integer

    # Find securities with any missing low/high prices and drop them
    incomplete_secs = secs.filter(pl.col("Low").is_null() | pl.col("High").is_null())["Security"].unique().to_list()
    secs = secs.filter(~pl.col("Security").is_in(incomplete_secs))
    if len(incomplete_secs) > 0:
        print(f"Dropping {len(incomplete_secs)} securities with missing low/high prices.")

    # Some securities lack open prices. In these cases, we set the open price to the previous close price.
    secs = secs.sort(["Security", "Date"]).with_columns(
        pl.when(pl.col("Open").is_null())
        .then(pl.col("Close").shift(1).over("Security"))  # group by Security
        .otherwise(pl.col("Open"))
        .alias("Open")
    )

    # Anonymize the securities (but sorted by the original issue ids)
    unique_iids = sorted(secs["Security"].unique().to_list())
    iid_map = {iid: f"{idx+1:04d}" for idx, iid in enumerate(unique_iids)}
    secs = secs.with_columns(secs["Security"].replace(iid_map).alias("Security"))

    # Re-compute the trading days after dropping securities
    secs = secs.with_columns(secs["Date"].unique().sort().search_sorted(secs["Date"]).alias("TradingDay"))

    # Save to parquet
    secs.write_parquet(output_path)
    print(f"Exported data for {len(unique_iids)} securities to {output_path}")


if __name__ == "__main__":

    root = amgm_config.dataset_root
    dset_path = root / "data-20250505"
    out_path = root / "securities_small.parquet"
    export_student_data(dset_path, out_path)
    dset_path = root / "data-20250901"
    out_path = root / "securities_large.parquet"
    export_student_data(dset_path, out_path)
