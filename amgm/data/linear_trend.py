import itertools
import logging
from datetime import datetime
from pathlib import Path
import random
from typing import Any, NamedTuple, Optional, Union, cast

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
import polars as pl
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from amgm import config as amgm_config
from amgm.data.base import BaseAMData, FeatureNormalizer
from amgm.data.loading import load_sebx_am_data
from amgm.utils import load_or_instantiate, common


class LinearTrendSample(NamedTuple):
    features: np.ndarray
    targets: np.ndarray
    issue_ids: Any
    test_dates: Any
    slopes: Any
    deviations: np.ndarray
    test_prices: Any
    ci_widths: Any
    ci_lowers: np.ndarray
    ci_uppers: np.ndarray
    margins: Any
    price_windows: np.ndarray


class LinearTrendDataset(BaseAMData, Dataset):

    def __init__(
        self,
        split: str,
        issue_ids: list[str],
        start_date: str,
        end_date_train: str,
        end_date_test: str,
        window_size_train=365,
        window_size_test=90,
        task_type="classification",
        label_variable="label_class",
        time_bar="daily",
        trend_value_col: str = "ClAdjLoc",
        feature_names: list[str] = ["ClAdjLoc"],
        normalization: dict[str, str] = None,
        balancing=None,                 # "None", "smallest", "residual-based", etc.
        balancing_bins=None,            # Optional callable (residuals: np.ndarray) -> dict[str, np.ndarray of indices], for residual-based balancing.
        balancing_target_count=None,    # Optional int, target samples per bin for residual-based balancing.
        output_range=None,              # Optional float, only used for residual-based balancing.
        training_data_type: str = "US_Stocks",  # "US_Stocks" or "RX1"
        data_source: str = "local",     # "local" or "yahoo"
        dset_path: Optional[Path] = None,   # It is set automatically for "US_Stocks" and "RX1" based on config, but must be set manually for other datasets.
        uploaded_df=None,
    ):
        """
        Args:
            split (str): Dataset split, 'train' or 'val'.
            issue_ids (list[str]): Issue IDs of the securities to include.

            start_date (str): Sampling start date in 'YYYY-MM-DD' format.
            end_date_train (str): End date for linear-trend fitting (+ network training) in 'YYYY-MM-DD' format.
            end_date_test (str): End date for linear-trend fit testing (+ label generation) in 'YYYY-MM-DD' format.
                The fit-testing start date is the end date of the linear-trend training set.

            window_size_test (int, optional): Number of trading days of features/trend-fitting per sample. Defaults to 365.
            time_bar (str, optional): Time bar frequency, 'daily', 'weekly', or 'monthly'. Defaults to "daily".
                Choosing any value other than "daily" will resample the data to the specified frequency.

            trend_value_col: (str): Name of the column to use for linear trend fitting. Defaults to "ClAdjLoc".
            feature_names (list[str]): Name of features to use in the samples. Defaults to ["ClAdjLoc"].
            normalization (dict[str, str], optional): Normalization method for each feature. Defaults to None.
            balancing (str, optional): Datasets balancing strategy, e.g., "smallest". Defaults to None.
            training_data_type (str, optional): The type of training data, either "US_Stocks" or "RX1". Defaults to "US_Stocks".
            data_source: str = "local" or "yahoo",
            dset_path: Optional[Path] = None,  # Only used if data_source is "local" and training_data_type is not "US Stocks" or "RX1"
            balance (bool, optional): Whether to balance the classes by subsampling. Defaults to True.
        """

        self.split = split
        self.time_bar = time_bar
        self.label_variable = label_variable
        self.balancing = balancing
        self.balancing_bins = balancing_bins
        if self.balancing == "residual-based":
            from trainer_cfg.regression_residual import resolve_balancing_bins
            self.balancing_bins = resolve_balancing_bins(self.balancing_bins)
        self.balancing_target_count = balancing_target_count
        self.window_size_train = window_size_train
        self.window_size_test = window_size_test

        self.excluded_iids = []

        if split == "train":
            start_date, end_date = start_date, end_date_train
        elif split == "val":
            start_date, end_date = end_date_train, end_date_test
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

        cls_name = self.__class__.__qualname__

        BaseAMData.__init__(
            self,
            issue_ids=issue_ids,
            start_date=start_date,
            end_date=end_date,
            feature_names=feature_names,
            normalization=normalization,
            log_name=cls_name,
            data_source=data_source,
            training_data_type=training_data_type,
            dset_path=dset_path,
            uploaded_df=uploaded_df,
        )

        secs = self.securities
        secs_by_iid = {iid: secs.filter(pl.col("IssueId") == iid) for iid in issue_ids}
     
        # Generate samples

        xs, ys, iids, test_dates, slopes, deviations, test_prices, ci_widths, ci_lowers, ci_uppers, margins, price_windows = [], [], [], [], [], [], [], [], [], [], [], []
        for iid in tqdm(issue_ids, desc=f"{cls_name}: creating {split} split"):
            if iid not in secs_by_iid:
                # Nothing in data for this ID, skip it
                self.excluded_iids.append(iid)
                continue
            x, y, test_date, slope, deviation, test_price, ci_width, ci_lower, ci_upper, margin, price_window = self.make_sample(
                secs_by_iid,
                iid,
                trend_value_col,
                feature_names,
                window_size_train,
                window_size_test,
            )
            xs.extend(x)
            ys.extend(y)
            iids.extend([iid] * len(x))
            test_dates.extend(test_date)
            slopes.extend(slope)
            deviations.extend(deviation)
            test_prices.extend(test_price)
            ci_widths.extend(ci_width)
            ci_lowers.extend(ci_lower)
            ci_uppers.extend(ci_upper)
            margins.extend(margin)
            price_windows.extend(price_window)

        print(f"Length of excluded Issue IDs: {len(self.excluded_iids)} for {split} split")
        
        assert len(xs) == len(ys)
        # Note: sklearn could possibly fail silently with non-contiguous arrays
        xs = np.ascontiguousarray(np.array(xs, dtype=np.float32))
        ys = np.ascontiguousarray(np.array(ys, dtype=np.float32))
        iids = np.array(iids, dtype=np.str_)
        test_dates = np.array(test_dates, dtype=np.str_)
        deviations = np.array(deviations, dtype=np.float32)
        slopes = np.array(slopes, dtype=np.float32)
        test_prices = np.array(test_prices, dtype=np.float32)
        ci_widths = np.array(ci_widths, dtype=np.float32)
        ci_lowers = np.array(ci_lowers, dtype=np.float32)
        ci_uppers = np.array(ci_uppers, dtype=np.float32)
        margins = np.array(margins, dtype=np.float32)
        price_windows = np.array(price_windows, dtype=np.float32)
        
        self.features = xs
        # Set target dtype based on task type
        if task_type == "classification":
            self.targets = ys.astype(int)
        else:  # regression
            self.targets = ys.astype(np.float32)
        self.issue_ids = iids
        self.test_dates = test_dates
        self.slopes = slopes
        self.deviations = deviations
        self.test_prices = test_prices
        self.ci_widths = ci_widths
        self.ci_lowers = ci_lowers
        self.ci_uppers = ci_uppers
        self.margins = margins
        self.price_windows = price_windows
        self.valid_features = np.isfinite(xs)
        self.valid_targets = np.isfinite(ys)

        self.log.warning(
            f"{len(self.excluded_iids)} issue IDs were excluded from the {split} split - insufficent data."
        )

        if balancing is not None:
            self.balanced_subsample_(self.balancing)

        if normalization is not None:
            self.normalize_features_(self.features, self.issue_ids, self.feature_names)

        self.log.info(f"{split.capitalize()} shape={self.features.shape}")

    def make_sample(self, 
                    secs_by_iid,
                    issue_id, 
                    trend_value_col, 
                    feat_cols, 
                    train_size, 
                    test_size, 
                    ):

        secs = secs_by_iid[issue_id]
        values = secs[trend_value_col].to_numpy()
        features = secs[feat_cols].to_numpy()
        dates_array = secs["Date"].dt.strftime("%Y-%m-%d").to_numpy()
        n = len(values)
        
        num_windows = n - train_size - test_size + 1

        if num_windows <= 0:
            self.excluded_iids += [issue_id]
            xs, ys, test_dates, slopes, deviations, test_prices, ci_widths, ci_lowers, ci_uppers, margins, price_windows = [], [], [], [], [], [], [], [], [], [], []
            return xs, ys, test_dates, slopes, deviations, test_prices, ci_widths, ci_lowers, ci_uppers, margins, price_windows

        num_features = features.shape[1]
        xs = np.empty((num_windows, train_size, num_features), dtype=features.dtype)
        ys = np.empty((num_windows, 1), dtype=np.float32)
        slopes = np.empty(num_windows, dtype=np.float32)
        deviations = np.empty((num_windows, test_size), dtype=np.float32) # deviation amount for each day in prediction period
        test_prices = np.empty(num_windows, dtype=np.float32)
        ci_widths = np.empty(num_windows, dtype=np.float32)
        test_dates = np.empty(num_windows, dtype=object)  # or keep as list
        price_windows = np.empty((num_windows, train_size + test_size), dtype=np.float32)
        
        ci_lowers = np.empty((num_windows, test_size), dtype=np.float32)
        ci_uppers = np.empty((num_windows, test_size), dtype=np.float32)
        margins = np.empty(num_windows, dtype=np.float32)
        
        for i, start in enumerate(range(0, num_windows)):
            test_idx = start + train_size - 1
            test_dates[i] = str(dates_array[test_idx])[:10] # Convert YYYY-MM-DDTHH:MM:SS.mmmmmm to YYYY-MM-DD
            xs[i] = features[start:test_idx + 1]
            output_dic = self.does_linear_trend_fit(values, test_idx, train_size, test_size, dates_array)
            # xs[i, :, 0] = residuals  # Replace price feature with its residuals from linear trend fit
            ys[i] = output_dic[self.label_variable]  # e.g., "label_class", "label_time"
            slopes[i] = output_dic["slope"]
            deviations[i] = output_dic["deviation_test_window"]
            test_prices[i] = output_dic["test_price"]
            ci_widths[i] = output_dic["ci_width_norm"]
            ci_lowers[i] = output_dic["lower_bound"]
            ci_uppers[i] = output_dic["upper_bound"]
            margins[i] = output_dic["margin"]
            price_windows[i] = output_dic["price_window"]

        return xs, ys, test_dates, slopes, deviations, test_prices, ci_widths, ci_lowers, ci_uppers, margins, price_windows

    def does_linear_trend_fit(
        self,
        values: np.ndarray,
        test_date_index: int,
        train_win_size: int,
        test_win_size: int,
        dates_array: None,
        CI_threshold: float = 1.96,
    ) -> dict:
        """Check whether a linear trend on the training window fits the values over the test window for a given confidence interval.

        Args:
            values (np.ndarray): Values to fit and test over
            test_date_index (int): Index of the testing date
            train_win_size (int): Size of the training window. [days]
            test_win_size (int): Size of the test window. [days]
            CI_threshold (float, optional): Threshold for confidence interval (default 1.96 for 95% CI).
            with_cofv (bool, optional): Whether to calculate the coefficient of variation of the slope. Defaults to True.

        Returns:
            dict: Dictionary containing various metrics and outputs from the linear trend fit.
        """

        output = {}
        x_index = np.arange(train_win_size + test_win_size)
        x_train = x_index[:train_win_size]
        y_train = values[test_date_index - train_win_size + 1: test_date_index + 1]

        # Resample to the time bar frequency
        d = dict(daily=1, weekly=5, monthly=21)[self.time_bar]

        y_train = y_train[::d]
        x_train = x_train[::d]
        slope, intercept = common.fast_linreg_numba(x_train, y_train)
        
        scale_fn = getattr(FeatureNormalizer, self.normalization.get("ClAdjLoc", ""), None)
        y_train_new = scale_fn(y_train[:, np.newaxis].T).flatten()
        slope_norm, intercept_norm = common.fast_linreg_numba(x_train, y_train_new)
        test_price = y_train[-1]  # Store the last price for later use

        # Predict on test window
        x_test = x_index[-test_win_size:]
        y_test = values[test_date_index + 1: test_date_index + test_win_size + 1]
    
        # Calculate predictions
        train_pred = intercept + slope * x_train
        test_pred = intercept + slope * x_test

        # Calculate confidence intervals using ±1.96 * std_err
        residuals = y_train - train_pred
        ci_width = CI_threshold * np.std(residuals)
        lower_bound = test_pred - ci_width
        upper_bound = test_pred + ci_width
        
        ci_width_norm = ci_width / (max(y_train) - min(y_train))  # Normalize CI width

        # Check coverage
        above = y_test > upper_bound
        below = y_test < lower_bound
        inside_ci = ~above & ~below
        coverage = np.mean(inside_ci)

        
        any_above = above.any()
        any_below = below.any()
        if inside_ci.all():
            label_class = 1
            label_time = test_win_size  # No crossing within the test window
        elif any_above and any_below:
            time_upper = np.where(above)[0][0]
            time_lower = np.where(below)[0][0]
            if time_upper < time_lower:
                label_class = 2  # Up crossing first
                label_time = time_upper
            else:
                label_class = 0  # Down crossing first
                label_time = time_lower
        elif any_above:
            label_class = 2
            label_time = np.where(above)[0][0]
        else:  # any_below:
            label_class = 0
            label_time = np.where(below)[0][0]

        
        deviation_test_window = np.zeros_like(y_test, dtype=float)
        mask = y_test >= upper_bound  
        deviation_test_window[mask] = 100 * (y_test[mask] - upper_bound[mask]) / upper_bound[mask]

        mask = y_test <= lower_bound
        deviation_test_window[mask] = 100 * (y_test[mask] - lower_bound[mask]) / lower_bound[mask]
        
        # clip deviations depending on the data
        if self.window_size_test == 63:  # For 3M prediction, we can have large deviations, so we clip at 50%
            deviation_test_window = np.clip(deviation_test_window, -50, 50)
        elif self.window_size_test == 5:  # For 5-day prediction, we can have smaller deviations, so we clip at 15%
            deviation_test_window = np.clip(deviation_test_window, -15, 15)
        max_deviation = deviation_test_window[np.argmax(np.abs(deviation_test_window))] 

        price_window = values[test_date_index - train_win_size + 1: test_date_index + test_win_size + 1]
        
        # Compute a single margin score for this sample based on the initial price
        # Margin = 1 - 2*|position - 0.5| where position is within CI [0, 1]
        # This gives 1 at center (hard to predict), 0 at edges (easy to predict)
        initial_price = y_test[0]
        if initial_price < lower_bound[0] or initial_price > upper_bound[0]:
            margin = 0.0  # Easy (outside CI)
        else:
            ci_range = upper_bound[0] - lower_bound[0]
            if ci_range > 0:
                position = (initial_price - lower_bound[0]) / ci_range
                margin = np.clip(1.0 - 2.0 * abs(position - 0.5), 0.0, 1.0)
            else:
                margin = 0.0
        
        output["label_class"] = label_class     # 0=down crossing first, 1=no crossing, 2=up crossing first
        output["label_time"] = (test_win_size - label_time) * (-1 if label_class == 0 else 1) # 0 if no crossing, positive(negative) if up(down) crossing occurs, higher means ealier
        output["label_residual"] = max_deviation
        output["coverage"] = coverage
        output["slope"] = slope_norm
        output["deviation_test_window"] = deviation_test_window
        output["test_price"] = test_price
        output["ci_width_norm"] = ci_width_norm
        output["lower_bound"] = lower_bound
        output["upper_bound"] = upper_bound
        output["margin"] = margin
        output["price_window"] = price_window
        
        return output

    def balanced_subsample(self, x, y, meta, strategy="smallest"):
        """Create the largest balanced subsample of a dataset

        Args:
            x (np.ndarray): Features of the dataset, shape (N, ...)
            y (np.ndarray): Labels of the dataset, shape (N, 1)
        Returns:
            Balanced subset of x and y, and balanced indices
        """

        self.log.info(f"Balancing {self.split} set with the {strategy} strategy.")

        if strategy == "smallest":
            unique_classes, counts = np.unique(y, return_counts=True)
            if len(counts) == 0:
                self.log.warning("No classes found in y; returning empty arrays.")
                # Return empty arrays with the correct shape
                return x[:0], y[:0], meta[:0], np.array([], dtype=int)
            min_count = np.min(counts)

            balanced_indices = []
            # Select min_count samples for each class
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                selected_indices = np.random.choice(class_indices, min_count, replace=False)
                balanced_indices.extend(selected_indices)

            balanced_indices = np.array(balanced_indices)
            return x[balanced_indices], y[balanced_indices], meta[balanced_indices], balanced_indices
        
        elif strategy == "residual-based":
            # Calculate average residual percentage for each y
            residuals = y.flatten()  # Assuming y is of shape (N, 1)
            # Define 11 bins based on residual percentage ranges

            bins = self.balancing_bins(residuals)
            target_count = self.balancing_target_count
            
            # Find the minimum count across all bins
            bin_counts = {name: len(indices) for name, indices in bins.items()}
            min_count = min(bin_counts.values())
            
            print(f"Min value of y (residuals): {residuals.min()}, max value of y: {residuals.max()}.")
            print(f"10 residuals randomly: {np.random.choice(residuals, 10, replace=False)}.")
            print(f"Bin counts before balancing: {bin_counts}, min_count={min_count}.")
            
            balanced_indices = []
            for bin_name, indices in bins.items():
                n_samples = len(indices)
                if n_samples == 0:
                    continue # Skip empty bins
                    
                # Oversample if we have too few, Undersample if we have too many
                replace = n_samples < target_count
                selected_indices = np.random.choice(indices, target_count, replace=replace)
                balanced_indices.extend(selected_indices)
            
            # Shuffle the combined indices
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            print(f"Len of balanced dataset: {len(balanced_indices)}, length of original dataset: {len(y)}.")
            
            return x[balanced_indices], y[balanced_indices], meta[balanced_indices], balanced_indices
        
        raise ValueError(f"Unknown balancing strategy: {strategy}. Supported: 'smallest', 'residual-based'.")

    def balanced_subsample_(self, strategy="smallest"):
        """Create the largest balanced subsample of the dataset in-place."""
        (self.features, self.targets, self.issue_ids, 
         balanced_indices) = self.balanced_subsample(
            self.features, self.targets, self.issue_ids, strategy
        )
        
        # Apply the same indices to all other attributes
        self.test_dates = self.test_dates[balanced_indices]
        self.slopes = self.slopes[balanced_indices]
        self.deviations = self.deviations[balanced_indices]
        self.test_prices = self.test_prices[balanced_indices]
        self.ci_widths = self.ci_widths[balanced_indices]
        self.ci_lowers = self.ci_lowers[balanced_indices]
        self.ci_uppers = self.ci_uppers[balanced_indices]
        self.margins = self.margins[balanced_indices]
        self.price_windows = self.price_windows[balanced_indices]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return LinearTrendSample(
            features=self.features[idx],
            targets=self.targets[idx],
            issue_ids=cast(Any, self.issue_ids)[idx],
            test_dates=self.test_dates[idx],
            slopes=self.slopes[idx],
            deviations=self.deviations[idx],
            test_prices=self.test_prices[idx],
            ci_widths=self.ci_widths[idx],
            ci_lowers=self.ci_lowers[idx],
            ci_uppers=self.ci_uppers[idx],
            margins=self.margins[idx],
            price_windows=self.price_windows[idx],
        )

class SectorIRMLinearTrendDataset(Dataset):
    """ Training-only dataset for invariant risk minimization training."""

    def __init__(self, *args, issue_ids: Optional[dict[str, list[int]]] = None, **kwargs):
        super().__init__()
        assert issue_ids is not None

        self.datasets = dict()
        for sec, iids in sorted(issue_ids.items(), key=lambda item: item[0]):
            self.datasets[sec] = LinearTrendDataset(*args, issue_ids=iids, **kwargs)

        ws = np.array([len(dset) for dset in self.datasets.values()])
        self.sector_weights =  ws / ws.sum()

    def __len__(self):
        return min(len(dset) for dset in self.datasets.values())

    def __getitem__(self, idx):
        # Pick random samples from each of the E datasets.
        samples = [dset[np.random.randint(len(dset))] for dset in self.datasets.values()]
        return samples


class LinearTrendDataModule(LightningDataModule):

    train_dataset: LinearTrendDataset
    val_dataset: LinearTrendDataset
    
    def __init__(
        self,
        run_cfg,
        dset_cfg,
        issue_ids: np.ndarray = None,   # If None, it will be either "RX1" or randomly sampled from dset_path for US_Stocks
        cache_path: Path = amgm_config.work_dir("linear_trend"),
        rebuild_cache=False,
        save_cache=True,
        build_train_dataset=True,
        uploaded_df=None,
    ):
        super().__init__()

        if dset_cfg["data_source"] == "local":
            if dset_cfg["training_data_type"] == "RX1":
                train_iids = ["RX1"]
                val_iids = ["RX1"]
            elif dset_cfg["training_data_type"] == "US_Stocks":
                num_iids = dset_cfg["num_iids"]
                data = load_sebx_am_data(dset_cfg["dset_path"])
                all_issue_ids = data["security_data"]["IssueId"].unique().tolist()
                
                if issue_ids is None:
                    issue_ids = np.array(random.sample(all_issue_ids, num_iids))  # Randomly sample without replacement
                    
                if issue_ids.shape[0] == 1:
                    print(f"Warning: Only one Issue ID sampled: {issue_ids[0]}")
                    train_iids = issue_ids.tolist()
                    val_iids = issue_ids.tolist()
                else:
                    # Split issue ids into train and validation sets
                    n = num_iids // 2
                    train_iids = issue_ids[:n].tolist()
                    val_iids = issue_ids[n:].tolist()
        elif dset_cfg["data_source"] == "yahoo":
            # For yfinance data, we assume issue_ids are provided directly as tickers
            train_iids = issue_ids.tolist()
            val_iids = issue_ids.tolist()
            
        dset_cfg = dset_cfg.copy()
        dset_cfg.pop("num_iids", None)
        dset_cfg.pop("split_method", None)

        # Handle balancing
        dset_cfg_train = dset_cfg.copy()
        dset_cfg_val = dset_cfg.copy()

        if dset_cfg.get("balancing", None) is not None:
            for split, strategy in dset_cfg["balancing"].items():
                if split == "train":
                    dset_cfg_train["balancing"] = strategy
                elif split == "val":
                    dset_cfg_val["balancing"] = strategy
                elif split == "test":
                    # Not used for now
                    continue
                else:
                    raise ValueError(f"Unknown split: {split}. Supported: 'train', 'val'.")
        
        train_dset_cfg = dict(_target_=LinearTrendDataset, 
                              split="train", 
                              issue_ids=train_iids, 
                              **dset_cfg_train)
        val_dset_cfg = dict(_target_=LinearTrendDataset, 
                            split="val", 
                            issue_ids=val_iids, 
                            **dset_cfg_val)

        self.run_cfg = run_cfg
        self.train_cfg = train_dset_cfg
        self.val_cfg = val_dset_cfg
        self.uploaded_df = uploaded_df
        self.cache_path = cache_path
        self.rebuild_cache = rebuild_cache
        self.save_cache = save_cache
        self.build_train_dataset = build_train_dataset

        batch_size = run_cfg["batch_size"]
        num_workers = run_cfg["num_workers"]

        self.loader_args = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
        )

    def prepare_data(self):
        if self.build_train_dataset:
            self.train_dataset, self.train_dataset_file = load_or_instantiate(
                self.train_cfg,
                "lt_train_dset",
                cache_dir=self.cache_path,
                rebuild=self.rebuild_cache,
                save=self.save_cache
            )

            assert np.isfinite(self.train_dataset.features).all()

        if self.uploaded_df is not None:
            # Uploaded validation data is runtime-only and not JSON-serializable for cache keys.
            val_kwargs = {k: v for k, v in self.val_cfg.items() if k != "_target_"}
            self.val_dataset = LinearTrendDataset(**val_kwargs, uploaded_df=self.uploaded_df)
            self.val_dataset_file = None
        else:
            self.val_dataset, self.val_dataset_file = load_or_instantiate(
                self.val_cfg,
                "lt_val_dset",
                cache_dir=self.cache_path,
                rebuild=self.rebuild_cache,
                save=self.save_cache
            )

    def worker_init_fn(worker_id):
        # Ensure each dataloader worker gets a different, reproducible seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.random.seed(worker_seed)

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            raise RuntimeError(
                "train_dataset is not prepared. "
                "Initialize LinearTrendDataModule with build_train_dataset=True for training."
            )
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_args)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_args)


class SectorIRMLinearTrendDataModule(LightningDataModule):
    """Data module for IRM (Invariant Risk Minimization) with linear trend features"""

    train_dataset: SectorIRMLinearTrendDataset
    val_dataset: LinearTrendDataset

    def __init__(
        self,
        run_cfg,
        dset_cfg,
        cache_path: Path = amgm_config.work_dir("linear_trend_irm"),
        rebuild_cache=False,
    ):
        super().__init__()

        data, iid2sec, sec2iids = LinearTrendDataset.load_data(dset_cfg["dset_path"])
        num_iids = dset_cfg["num_iids"]

        # Uniformly sample issue ids from each sector assigning half for training and half for validation.

        train_iids = dict()
        val_iids = dict()

        n_sectors = len(sec2iids)
        iids_per_sector = int(num_iids / n_sectors + 0.5)
        for sec, sec_iids in sec2iids.items():
            iids = np.random.choice(sec_iids, min(iids_per_sector, len(sec_iids)), replace=False).tolist()
            i = len(iids) // 2
            train_iids[sec] = iids[:i]
            val_iids[sec] = iids[i:]

        example_iids = next(iter(train_iids.values()))[:10]
        
        dset_cfg = dset_cfg.copy()
        dset_cfg.pop("num_iids", None)
        dset_cfg.pop("split_method", None)

        # Handle balancing
        dset_cfg_train = dset_cfg.copy()
        dset_cfg_val = dset_cfg.copy()

        if dset_cfg.get("balancing", None) is not None:
            for split, strategy in dset_cfg["balancing"].items():
                if split == "train":
                    dset_cfg_train["balancing"] = strategy
                elif split == "val":
                    dset_cfg_val["balancing"] = strategy
                elif split == "test":
                    # Not used for now
                    continue
                else:
                    raise ValueError(f"Unknown split: {split}. Supported: 'train', 'val'.")

        train_dset_cfg = dict(_target_=SectorIRMLinearTrendDataset, split="train", issue_ids=train_iids, **dset_cfg_train)
        val_iids = list(itertools.chain(*val_iids.values()))  # -> flat list of iids from dictionary of (str, list)
        val_dset_cfg = dict(_target_=LinearTrendDataset, split="val", issue_ids=val_iids, **dset_cfg_val)

        self.run_cfg = run_cfg
        self.train_cfg = train_dset_cfg
        self.val_cfg = val_dset_cfg
        self.cache_path = cache_path
        self.rebuild_cache = rebuild_cache

        batch_size = run_cfg["batch_size"]
        num_workers = run_cfg["num_workers"]

        self.loader_args = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
        )

    def prepare_data(self):
        self.train_dataset, self.train_dataset_file = load_or_instantiate(
            self.train_cfg,
            "sec_irm_lt_train_dset",
            cache_dir=self.cache_path,
            rebuild=self.rebuild_cache,
        )
        self.val_dataset, self.val_dataset_file = load_or_instantiate(
            self.val_cfg,
            "sec_irm_lt_val_dset",
            cache_dir=self.cache_path,
            rebuild=self.rebuild_cache,
        )

        for sec, dset in self.train_dataset.datasets.items():
            assert np.isfinite(dset.features).all(), f"sector {sec} features has non-finite values"

        return

    def worker_init_fn(worker_id):
        # Ensure each dataloader worker gets a different, reproducible seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.random.seed(worker_seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self._collate, shuffle=True, **self.loader_args)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_args)

    @staticmethod
    def _collate(batch):
        from torch.utils.data._utils.collate import collate, default_collate_fn_map

        B = len(batch)  # Batch size
        E = len(batch[0])  # Number of IRM environments
        S = len(batch[0][0])  # Number of tuple elements in a sample

        # Reorganize (B, E, S) -> (S, B, E)
        batch2 = [np.array([[batch[b][e][s] for e in range(E)] for b in range(B)]) for s in range(S)]

        # Cast to batch output format
        for s in range(S):
            sample = batch2[s]
            if np.issubdtype(sample.dtype, np.number):
                batch2[s] = torch.from_numpy(sample)
            else:
                # Strings have no torch dtype, and cannot be stored in tensors
                batch2[s] = sample.tolist()

        return batch2
