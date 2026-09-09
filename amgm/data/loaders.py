"""Dataset loaders for `BaseAMData`.

Each loader is a small class that knows how to read ONE kind of dataset and
return a polars DataFrame with at least these columns:
    - "IssueId"  (string)
    - "Date"     (date)
    - "ClAdjLoc" (float)   -- plus any other columns your features need.

How to add a new dataset (for example, a Bloomberg "US Stocks_top100" file):

    1. Add a new class below that inherits from `DataLoader` and implements `load()`.
       Example:

           class BloombergTop100Loader(DataLoader):
               def load(self, issue_ids, start_date, end_date):
                   df = pd.read_parquet(self.dset_path)
                   # ... reshape df so it has IssueId, Date, ClAdjLoc ...
                   return pl.from_pandas(df)

    2. Add one more `elif` branch in `make_loader()` below to wire it up:

           elif data_source == "Bloomberg" and training_data_type == "US_Stocks_top100":
               return BloombergTop100Loader(dset_path)

    3. Use it from BaseAMData exactly like the built-in ones:

           BaseAMData(
               issue_ids=[...],
               start_date="2020-01-01", end_date="2024-12-31",
               data_source="Bloomberg",
               training_data_type="US_Stocks_top100",
               dset_path=CUSTOM_PATH_TO_DS,
           )
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import yfinance as yf
from amgm import config as amgm_config
from amgm.data.loading import load_sebx_am_data
        
class DataLoader:
    """Base class for dataset loaders. Subclass it and override `load()`."""

    def __init__(self, dset_path: Optional[Path] = None):
        self.dset_path = dset_path

    def load(self, issue_ids: list[str], start_date: str, end_date: str) -> pl.DataFrame:
        raise NotImplementedError("Subclasses must implement load().")


# ---------------------------------------------------------------------------
# Built-in loaders
# ---------------------------------------------------------------------------


class USStocksLoader(DataLoader):
    """AM US_Stocks dataset from local files."""

    def load(self, issue_ids, start_date, end_date):
        path = amgm_config.am_dataset_dir
        data = load_sebx_am_data(path, cached=True, force=False)
        secs = pl.from_pandas(data["security_data"])
        # cast IssueId to string to avoid performance issues
        secs = secs.with_columns(pl.col("IssueId").cast(pl.Utf8))
        return secs[["IssueId", "Date", "ClAdjLoc"]]


class RX1Loader(DataLoader):
    """RX1 Bund futures from a two-header-row CSV."""

    def load(self, issue_ids, start_date, end_date):
        path = amgm_config.rx1_dataset_dir
        # The file has two header rows; row 2 contains Bloomberg tickers (including RX1 Comdty).
        df = pd.read_csv(path, header=1)
        date_col = df.columns[0]

        df = df[[date_col, "RX1 Comdty"]].copy()
        df = df.rename(columns={date_col: "Date", "RX1 Comdty": "ClAdjLoc"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["ClAdjLoc"] = pd.to_numeric(df["ClAdjLoc"], errors="coerce")
        df = df.dropna(subset=["Date", "ClAdjLoc"])
        df["Date"] = df["Date"].dt.date
        df["IssueId"] = "RX1"
        return pl.from_pandas(df[["IssueId", "Date", "ClAdjLoc"]])


class YahooLoader(DataLoader):
    """Fetch OHLC data from Yahoo Finance via yfinance."""

    def load(self, issue_ids, start_date, end_date):
        end_plus_one = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frames = []
        for ticker in issue_ids:
            print(
                f"[DEBUG][YahooLoader] start ticker={ticker} start_date={start_date} end_date={end_date} end_plus_one={end_plus_one}"
            )
            df = yf.download(ticker, start=start_date, end=end_plus_one, auto_adjust=True)
            if df is None:
                raise ValueError(f"[DEBUG][YahooLoader] yf.download returned None for ticker={ticker}")

            raw_columns = [str(c) for c in list(df.columns)]
            print(
                "[DEBUG][YahooLoader] "
                f"ticker={ticker} raw_shape={df.shape} raw_columns={raw_columns} "
                f"index_name={getattr(df.index, 'name', None)} index_type={type(df.index).__name__}"
            )

            if df.empty:
                print(f"[DEBUG][YahooLoader] ticker={ticker} returned empty dataframe from yfinance")

            df.columns = ["_".join([str(i) for i in col if i]) for col in df.columns]
            df = df.reset_index()
            print(
                f"[DEBUG][YahooLoader] ticker={ticker} columns_after_reset={list(df.columns)} shape_after_reset={df.shape}"
            )

            # yfinance can return an unnamed DatetimeIndex on some environments
            # (e.g., container builds), which becomes "index" after reset_index().
            if "Date" not in df.columns:
                date_candidates = [
                    c
                    for c in df.columns
                    if str(c).lower() in {"index", "datetime", "date"}
                    or "date" in str(c).lower()
                    or "time" in str(c).lower()
                ]
                if date_candidates:
                    chosen_date_col = date_candidates[0]
                    df = df.rename(columns={chosen_date_col: "Date"})
                    print(
                        f"[DEBUG][YahooLoader] ticker={ticker} renamed date column {chosen_date_col} -> Date"
                    )

            close_col = f"Close_{ticker}"
            if close_col not in df.columns:
                close_candidates = [c for c in df.columns if str(c).startswith("Close")]
                print(
                    f"[DEBUG][YahooLoader] ticker={ticker} expected_close_col={close_col} "
                    f"close_candidates={close_candidates}"
                )
                if not close_candidates:
                    raise KeyError(
                        "[DEBUG][YahooLoader] Missing close column after flattening: "
                        f"ticker={ticker}, columns={list(df.columns)}"
                    )
                close_col = close_candidates[0]

            df["ClAdjLoc"] = df[close_col]
            df["IssueId"] = ticker

            if "Date" not in df.columns:
                candidate_date_cols = [c for c in df.columns if "date" in str(c).lower() or str(c).lower() == "index"]
                debug_sample = df.head(3).to_dict(orient="records")
                raise KeyError(
                    "[DEBUG][YahooLoader] Missing Date column after reset_index: "
                    f"ticker={ticker}, columns={list(df.columns)}, candidate_date_cols={candidate_date_cols}, "
                    f"index_name_before_reset={getattr(df.index, 'name', None)}, sample={debug_sample}"
                )

            frames.append(df[["IssueId", "Date", "ClAdjLoc"]])

        if not frames:
            raise ValueError(
                "[DEBUG][YahooLoader] No frames were built. "
                f"issue_ids={issue_ids}, start_date={start_date}, end_date={end_date}"
            )

        print(f"[DEBUG][YahooLoader] concatenating {len(frames)} frame(s)")
        return pl.from_pandas(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# Factory: pick the right loader for the given configuration.
# To add a new dataset, just add another `elif` branch here.
# ---------------------------------------------------------------------------


def make_loader(
    data_source: str,
    training_data_type: str,
    dset_path: Optional[Path] = None,
) -> DataLoader:
    if data_source == "local" and training_data_type == "US_Stocks":
        return USStocksLoader(dset_path)

    elif data_source == "local" and training_data_type == "RX1":
        return RX1Loader(dset_path)

    elif data_source == "yahoo":
        return YahooLoader(dset_path)

    # Add your own dataset here, for example:
    # elif data_source == "Bloomberg" and training_data_type == "US_Stocks_top100":
    #     return BloombergTop100Loader(dset_path)

    else:
        raise ValueError(
            f"No loader for data_source={data_source!r}, training_data_type={training_data_type!r}. "
            "Add a new class in loaders.py and a new branch in make_loader()."
        )
