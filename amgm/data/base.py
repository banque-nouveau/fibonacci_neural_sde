from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch

from amgm.data.loaders import make_loader


class FeatureNormalizer:

    @classmethod
    def sample_minmax(cls, X: np.ndarray, groups=None) -> np.ndarray:
        x_min, x_max = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
        return (X - x_min) / (x_max - x_min + 1e-6)

    @classmethod
    def sample_zscore(cls, X: np.ndarray, groups=None) -> np.ndarray:
        x_mean, x_std = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
        return (X - x_mean) / (x_std + 1e-6)

    @classmethod
    def group_minmax(cls, X: np.ndarray, groups) -> np.ndarray:
        for group in np.unique(groups):
            mask = groups == group
            X[mask] = cls.sample_minmax(X[mask])
            assert np.isfinite(X[mask]).all()
        return X

    @classmethod
    def group_zscore(cls, X: np.ndarray, groups) -> np.ndarray:
        for group in np.unique(groups):
            mask = groups == group
            X[mask] = cls.sample_zscore(X[mask])
            assert np.isfinite(X[mask]).all()
        return X

    @classmethod
    def div_100(cls, X: np.ndarray, groups) -> np.ndarray:
        return X / 100.0


class BaseAMData:
    """AM base dataset, providing common functionality for all dataset use cases."""

    def __init__(
        self,
        issue_ids: list[str],
        start_date: str,
        end_date: str,
        feature_names: list[str] = ["ClAdjLoc"],
        normalization: dict[str, str] = None,
        log_name=None,
        data_source: str = "local",
        training_data_type: str = "US_Stocks",
        dset_path: Optional[Path] = None,
        uploaded_df=None,
    ):
        """
        Args:
            issue_ids (list[str]): Issue IDs of the securities to form samples from.
            start_date (str): Sampling start date in 'YYYY-MM-DD' format.
            end_date (str): Sampling end date in 'YYYY-MM-DD' format.

            feature_names (list[str]): Name of features to use in the samples. Defaults to ["ClAdjLoc"].
            normalization (dict[str, str], optional): Normalization method for each feature. Defaults to None.
            log_name (str, optional): Name of the logger. Defaults to None, which uses the module name.
            data_source (str, optional): Where to load data from, "local" or "yahoo". Defaults to "local".
            training_data_type (str, optional): "US_Stocks" or "RX1". Defaults to "US_Stocks".
            dset_path (Path, optional): Path to the dataset root directory. It is set automatically for "US_Stocks" and "RX1" based on config, but can be set manually for other data sources. Defaults to None.
        """

        self.log = logging.getLogger(name=log_name)
        self.feature_names = feature_names
        self.normalization = normalization

        self.excluded_iids = []

        secs = self.load_data(
            training_data_type,
            issue_ids,
            start_date,
            end_date,
            data_source,
            dset_path,
            uploaded_df=uploaded_df,
        )

        secs = self.compute_trading_days(secs)
        self.log.info(f"Adding features: {feature_names}")
        secs = self.add_features(secs, feature_names)

        d1 = datetime.strptime(start_date, "%Y-%m-%d").date()
        d2 = datetime.strptime(end_date, "%Y-%m-%d").date()

        secs = secs.filter(pl.col("IssueId").is_in(issue_ids) & (pl.col("Date") >= d1) & (pl.col("Date") <= d2))
        self.securities = secs
    
    @classmethod
    def load_data(
        cls,
        training_data_type,
        issue_ids,
        start_date,
        end_date,
        data_source="local",
        dset_path=None,
        uploaded_df=None,
    ):
        """Load data via the loader registered for (data_source, training_data_type).

        Returns:
            pl.DataFrame with columns "IssueId", "Date", and "ClAdjLoc"
        """
        if uploaded_df is not None:
            secs = uploaded_df if isinstance(uploaded_df, pl.DataFrame) else pl.from_pandas(uploaded_df)

            if "IssueId" not in secs.columns and "ticker" in secs.columns:
                secs = secs.rename({"ticker": "IssueId"})

            required_cols = {"IssueId", "Date", "ClAdjLoc"}
            missing_cols = sorted(required_cols - set(secs.columns))
            if missing_cols:
                raise ValueError(
                    f"Uploaded dataframe is missing required column(s): {missing_cols}. "
                    "Expected columns include IssueId/Date/ClAdjLoc."
                )

            if secs.schema.get("Date") != pl.Date:
                secs = secs.with_columns(pl.col("Date").cast(pl.Date, strict=False))

            return secs

        loader = make_loader(data_source, training_data_type, dset_path)
        return loader.load(issue_ids, start_date, end_date)

    @staticmethod
    def compute_trading_days(secs: pl.DataFrame) -> pl.DataFrame:
        """Add a column "TradingDay" from the "Date" column, by counting the number of trading days since the beginning,
        starting from zero. The count skips days where no trading occurred, so it is not a calendar day count.
        """
        secs = secs.with_columns(secs["Date"].unique().sort().search_sorted(secs["Date"]).alias("TradingDay"))
        return secs

    @staticmethod
    def find_gaps(secs: pl.DataFrame) -> pl.DataFrame:
        """Find and add a column marking gaps in the trading days of each security, i.e.,
        TradingDay where no trading occurred, per IssueId."""
        assert "TradingDay" in secs.columns, "TradingDay column is required to find gaps."
        d = secs.sort(["IssueId", "TradingDay"])
        d = d.with_columns([pl.col("TradingDay").over("IssueId").shift(1).alias("prev_day")])
        gaps = d.with_columns([(pl.col("TradingDay").cast(int) - pl.col("prev_day") - 1).alias("gap")])
        return gaps

    @classmethod
    def find_one_day_gaps(cls, secs: pl.DataFrame) -> pl.DataFrame:
        """Find one-day gaps in the trading days of each security, to be masked or filled in."""
        d = cls.find_gaps(secs)
        d = d.with_columns([(pl.col("TradingDay") - 1).alias("missing_day")])
        gaps = d.filter(pl.col("gap") == 1)[["IssueId", "missing_day", "prev_day"]]
        return gaps

    @staticmethod
    def add_features(secs: pl.DataFrame, feature_names: list[str]) -> pl.DataFrame:
        """ Add price features to the dataset, based on the specified feature names. The features are added in the order of the feature_names list."""
        
        from amgm.data.price_features import (
            add_clvol,
            add_hurst,
            add_sma_crossover,
            add_min_1d_return,
            add_max_1d_return,
            add_cumulative_return,
            add_pctbb,
            add_rsi14,
            add_obv,
            add_log_value,
            add_sum
        )

        for name in feature_names:
            if name in secs.columns:
                continue
            # price features
            elif name == "Cl1mVol":
                secs = add_clvol(secs, src="ClAdjUsd", dst=name)
            elif name == "Hurst1m":
                secs = add_hurst(secs, src="ClAdjUsd", dst=name)
            elif name == "Sma50x200":
                secs = add_sma_crossover(secs, src="ClAdjUsd", dst=name, short_win=50, long_win=200)
            elif name == "Sma50x100":
                secs = add_sma_crossover(secs, src="ClAdjUsd", dst=name, short_win=50, long_win=100)
            elif name == "Sma100x200":
                secs = add_sma_crossover(secs, src="ClAdjUsd", dst=name, short_win=100, long_win=200)
            elif name == "Min1dRet1m":
                secs = add_min_1d_return(secs, src="ClAdjUsd", dst=name)
            elif name == "Max1dRet1m":
                secs = add_max_1d_return(secs, src="ClAdjUsd", dst=name)
            elif name == "CumRet1m":
                secs = add_cumulative_return(secs, src="ClAdjUsd", dst=name)
            elif name == "PctBb":
                secs = add_pctbb(secs, src="ClAdjUsd", dst=name)
            elif name == "Rsi14":
                secs = add_rsi14(secs, src="ClAdjUsd", dst=name)
            elif name == "OpClAdjUsdSum":
                # Sum of open and close prices
                secs = secs.with_columns([(pl.col("OpAdjUsd") + pl.col("ClAdjUsd")).alias(name)])
            # Volume features
            elif name == "OnBalanceVolume":
                secs = add_obv(secs, price_col="ClAdjUsd", volume_col="VoAdj", dst=name)
            elif name == "SumOpClAdjLoc":
                secs = add_sum(secs, srcs=["OpAdjLoc", "ClAdjLoc"], dst=name)
            elif name == "LogClAdjUsd":
                secs = add_log_value(secs, src="ClAdjUsd", dst=name)
            elif name == "LogClAdjLoc":
                secs = add_log_value(secs, src="ClAdjLoc", dst=name)
            else:
                raise ValueError(f"Unknown feature name: {name}")

        return secs

    @staticmethod
    def normalize_column(secs: pl.DataFrame, src: str, grp: str, dst: str, type: str):
        """Normalize the values of a column, grouped by some other column, e.g "IssueId".
        Args:
            secs (pl.DataFrame): The securities to normalize.
            src (str): The column to normalize. (Source)
            grp (str): A column to group by (e.g IssueId). Each group is normalized separately.
            dst (str): The output column name for the normalized values. (Destination)
            type (str): The normalization type, either "zscore" or "minmax". Default is "minmax".
        Returns:
            pL.DataFrame: The normalized securities in the dst column.
        """

        d = pl.col(src)
        eps = 1e-6  # Small epsilon to avoid division by zero

        if type == "zscore":
            a = d.mean().over(grp)
            b = d.std().over(grp)
            b = pl.when(b == 0).then(eps).otherwise(b)

        elif type == "minmax":
            a = d.min().over(grp)
            b = d.max().over(grp) - a
            b = pl.when(b == 0).then(eps).otherwise(b)

        elif type == "div_100":
            a, b = 0, 100

        else:
            raise ValueError(f"Unknown normalization type: {type}. Use 'zscore', 'minmax', or 'div_100'.")

        # Sorting does not work. Polars might reorder the rows so that the normalization fails (?):
        # secs = secs.sort([col, "TradingDay"])
        secs = secs.with_columns(((d - a) / b).alias(dst))
        return secs

    def normalize_features_(self, features: torch.Tensor, group_ids: list[str], feature_names: list[str]):
        """Normalize the features of the dataset when organized in a tensor, over groups of samples

        Args:
            features (torch.Tensor): Features, shape (N, ..., C)
                holding N samples, each with C features (or a C-dimensional feature vector)
            group_ids (list[str]): List of N group IDs, one per sample.
                IDs of the same group are normalized together, if grouped normalization is applied.
            feature_names (list[str]): List of C feature names, one per feature
        """
        if self.normalization is None:
            return

        # Early exit if features are empty
        if features.shape[0] == 0:
            self.log.warning("No features to normalize (empty array). Skipping normalization.")
            return

        for i, name in enumerate(feature_names):
            scale_fn = getattr(FeatureNormalizer, self.normalization.get(name, ""), None)
            if scale_fn is None:
                continue
            features[..., i] = scale_fn(features[..., i], group_ids)
            assert np.isfinite(features[..., i]).all(), f"Normalization failed for {name}."

    def visualize_column(self, secs, col: str):
        """Visualize the specified column of the dataset as an image. The x-axis represents the IssueId, and the y-axis
        represents the TradingDay. The intensity of the color represents the minmax normalized value of the specified column.

        Args:
            secs (pl.DataFrame): Security data DataFrame with columns "IssueId", "TradingDay", and the specified column.
            col (str): The name of the column to visualize, for exmple, "ClAdjLoc".
            im_file (Union[Path, str], optional): The name of the output image file. Defaults to "securities_visualization.png".
        """

        from matplotlib import pyplot as plt

        cmap = plt.get_cmap("Paired")
        secs = self.securities

        assert "TradingDay" in secs.columns

        # Map to colors and x,y coordinates

        secs = secs.sort(["IssueId", "TradingDay"])

        iids = secs["IssueId"].unique(maintain_order=True)
        iid2idx = {iid: i for i, iid in enumerate(iids.to_list())}  # don't sort
        secs = secs.with_columns(pl.col("IssueId").cast(pl.Utf8))  # -> str to avoid performance issues
        secs = secs.with_columns(pl.col("IssueId").replace_strict(iid2idx).alias("IssueIdx"))  # x-coordinate

        gaps = self.find_one_day_gaps(secs)
        gaps = gaps.with_columns(pl.col("IssueId").replace_strict(iid2idx).alias("IssueIdx"))
        gap_xs = gaps["IssueIdx"].to_numpy()  # gap x-coordinates
        gap_ys = gaps["missing_day"].to_numpy()  # gap y-coordinates
        i = np.where(gap_ys >= 0)
        gap_xs, gap_ys = gap_xs[i], gap_ys[i]

        colors = np.array([0, 255, 0], dtype=np.uint8)  # Default color for all securities
        # colors = (cmap(secs["SectorIdx"]) * 255).astype(np.uint8)[:, :3]

        xs = secs["IssueIdx"].to_numpy()  # x-coordinates
        ys = secs["TradingDay"].to_numpy()  # y-coordinates

        src = col
        dst = src + "_n"
        cdata = self.normalize_column(secs, src=src, grp="IssueId", dst=dst, type="minmax")[dst].to_numpy()

        # Paint the image
        num_securities = secs["IssueId"].n_unique()
        num_days = secs["TradingDay"].max() + 1
        im = np.zeros((num_days, num_securities, 3), dtype=np.uint8)

        im[:, :] = np.array([64, 64, 64], dtype=np.uint8)  # Background color
        im[ys, xs] = ((colors * cdata[:, np.newaxis])).clip(0, 255).astype(np.uint8)
        im[gap_ys, gap_xs] = np.array([255, 0, 255], dtype=np.uint8)

        return im
