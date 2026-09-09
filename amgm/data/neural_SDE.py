from pathlib import Path
from typing import Any, NamedTuple, Optional
import random
from amgm import config as amgm_config

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from amgm.data.loading import load_sebx_am_data
from tqdm import tqdm

from amgm.data.base import BaseAMData
import amgm.utils.common as common

class NeuralSDESample(NamedTuple):
    price_window: torch.Tensor
    nxt_price: torch.Tensor
    sample_min: torch.Tensor
    sample_range: torch.Tensor
    features: torch.Tensor
    fib_levels: torch.Tensor
    test_dates: Any
    issue_ids: Any

class NeuralSDEDataset(BaseAMData, Dataset):

    def __init__(
        self,
        issue_ids: Optional[list[str]],
        num_iids: int,      # Only used if issue_ids is None or empty, otherwise ignored
        start_date: str,
        end_date: str,
        lookback_window = 365,
        max_windows_per_issue: Optional[int] = None,
        dt = 1.0,
        price_value_col: str = "ClAdjLoc",
        training_data_type: str = "US_Stocks",  # "US_Stocks" or "RX1"
        data_source: str = "local",     # "local" or "yahoo"
        dset_path = amgm_config.am_dataset_dir,   # It is set automatically for "US_Stocks" and "RX1" based on config, but must be set manually for other datasets.
        uploaded_df = None,
        rng_seed: Optional[int] = None,
    ):
        """
        Args:
            issue_ids (list[str]): Issue IDs of the securities to include.
            start_date (str): Sampling start date in 'YYYY-MM-DD' format.
            end_date (str): End date for linear-trend fitting (+ network training) in 'YYYY-MM-DD' format.
            lookback_window (int, optional): Number of trading days per sample. Defaults to 365.
            price_value_col: (str): Name of the column to use for price data. Defaults to "ClAdjLoc".
            training_data_type (str, optional): The type of training data, either "US_Stocks" or "RX1". Defaults to "US_Stocks".
            data_source: str = "local" or "yahoo",
            dset_path: Optional[Path] = None,  # Only used if data_source is "local" and training_data_type is not "US Stocks" or "RX1"
        """

        self.lookback_window = lookback_window
        self.max_windows_per_issue = max_windows_per_issue
        self.excluded_iids = []
        cls_name = self.__class__.__qualname__

        if issue_ids is None or len(issue_ids) == 0:
            # Randomly pick num_iids issue IDs from dset_path if issue_ids is not provided
            data = load_sebx_am_data(dset_path)
            all_issue_ids = data["security_data"]["IssueId"].unique().tolist()
            rng = random.Random(rng_seed)
            issue_ids = rng.sample(all_issue_ids, num_iids)
        
        print(f"Selected {len(issue_ids)} issue IDs for dataset: {issue_ids}")
        
        BaseAMData.__init__(
            self,
            issue_ids=issue_ids,
            start_date=start_date,
            end_date=end_date,
            feature_names=[price_value_col],
            normalization={},
            log_name=cls_name,
            data_source=data_source,
            training_data_type=training_data_type,
            dset_path=dset_path,
            uploaded_df=uploaded_df,
        )

        secs = self.securities
        secs_by_iid = {iid: secs.filter(pl.col("IssueId") == iid) for iid in issue_ids}
     
        # Generate samples
        price_windows, nxt_prices, test_dates, iids = [], [], [], []
        for iid in tqdm(issue_ids, desc=f"{cls_name}: creating samples"):
            if iid not in secs_by_iid:
                # Nothing in data for this ID, skip it
                self.excluded_iids.append(iid)
                continue
            price_window, nxt_price, test_date = self.make_sample(
                secs_by_iid,
                iid,
                price_value_col,
                lookback_window
            )
            price_windows.extend(price_window)
            nxt_prices.extend(nxt_price)
            test_dates.extend(test_date)
            iids.extend([iid] * len(price_window))

        print(f"Length of excluded Issue IDs: {len(self.excluded_iids)}")
        
        price_windows = np.ascontiguousarray(np.array(price_windows, dtype=np.float32))
        nxt_prices = np.ascontiguousarray(np.array(nxt_prices, dtype=np.float32))
        test_dates = np.array(test_dates, dtype=np.str_)

        # Per-sample min-max normalization for each price window.
        price_windows, nxt_prices, sample_min, sample_range = self._min_max_normalize(price_windows, nxt_prices)
        
        # Calculate features after normalizations
        features, fib_levels = self.calculate_features(price_windows)
        
        print(f"Price windows shape: {price_windows.shape}, Next prices shape: {nxt_prices.shape}, Features shape: {features.shape}")
        assert len(price_windows) == len(features)
        
        self.price_window = torch.from_numpy(price_windows)
        self.nxt_prices = torch.from_numpy(nxt_prices)
        self.sample_min = torch.from_numpy(sample_min)
        self.sample_range = torch.from_numpy(sample_range)
        self.features = torch.from_numpy(features)
        self.fib_levels = torch.from_numpy(fib_levels)
        self.test_dates = test_dates
        self.issue_ids = iids
        
        self.log.warning(
            f"{len(self.excluded_iids)} issue IDs were excluded - insufficient data."
        )

        self.log.info(f"Features shape after normalization={self.features.shape}")

    def make_sample(self, 
                    secs_by_iid,
                    issue_id, 
                    price_value_col, 
                    lookback_window,
                    ):

        secs = secs_by_iid[issue_id]
        values = secs[price_value_col].to_numpy()
        dates_array = secs["Date"].dt.strftime("%Y-%m-%d").to_numpy()
        n = len(values)
        
        num_windows = n - (lookback_window + 1) + 1     # lookback_window + 1 to include the next day price for prediction
        if self.max_windows_per_issue is not None:
            num_windows = min(num_windows, self.max_windows_per_issue)

        if num_windows <= 0:
            self.excluded_iids += [issue_id]
            xs, nxt_prices, test_dates = [], [], []
            return xs, nxt_prices, test_dates
        
        price_windows = np.empty((num_windows, lookback_window), dtype=values.dtype)
        nxt_prices = np.empty((num_windows, 1), dtype=values.dtype)
        test_dates = np.empty(num_windows, dtype=object)  # or keep as list
        
        for i, start in enumerate(range(0, num_windows)):
            test_idx = start + lookback_window - 1
            test_dates[i] = str(dates_array[test_idx])[:10] # Convert YYYY-MM-DDTHH:MM:SS.mmmmmm to YYYY-MM-DD
            price_windows[i] = values[start:test_idx + 1]
            nxt_prices[i] = values[test_idx + 1]
        return price_windows, nxt_prices, test_dates

    def calculate_features(self, price_windows):
        """Calculate features for each price window, such as Fibonacci levels."""
        
        fib_levels, delta = common.calculate_fibLevels(price_windows)
        last_price = price_windows[:, -1:]
        features = (last_price - fib_levels) / np.maximum(delta, 1e-8)
        
        return features, fib_levels

    @staticmethod
    def _min_max_normalize(price_windows, nxt_prices):
        """Per-sample min-max normalization for each price window."""
        pw_min = price_windows.min(axis=1, keepdims=True)
        pw_max = price_windows.max(axis=1, keepdims=True)
        pw_range = np.maximum(pw_max - pw_min, 1e-8)
        price_windows = (price_windows - pw_min) / pw_range
        nxt_prices = (nxt_prices - pw_min) / pw_range
        return price_windows, nxt_prices, pw_min.astype(np.float32), pw_range.astype(np.float32)
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return NeuralSDESample(
            price_window=self.price_window[idx],
            nxt_price=self.nxt_prices[idx],
            sample_min=self.sample_min[idx],
            sample_range=self.sample_range[idx],
            features=self.features[idx],
            fib_levels=self.fib_levels[idx],
            test_dates=self.test_dates[idx],
            issue_ids=self.issue_ids[idx]
        )
        

class SyntheticNeuralSDEDataset(Dataset):
    """Synthetic one-dimensional SDE samples with known drift/diffusion.

    SDE used for simulation:
        dX_t = kappa * (theta - X_t) dt + (sigma_base + sigma_scale * sigmoid(X_t)) dW_t
    """

    def __init__(
        self,
        lookback_window,
        dt=1.0 / 252.0,
        num_paths=256,
        steps_per_path=64,
        x0_mean=0.0,
        x0_std=0.2,
        kappa=2.0,
        theta=0.0,
        sigma_base=0.05,
        sigma_scale=0.15,
        rng_seed=1,
        **_,
    ):
        self.lookback_window = int(lookback_window)
        self.dt = float(dt)
        self.num_paths = int(num_paths)
        self.steps_per_path = int(steps_per_path)
        self.x0_mean = float(x0_mean)
        self.x0_std = float(x0_std)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_base = float(sigma_base)
        self.sigma_scale = float(sigma_scale)

        gen = torch.Generator().manual_seed(int(rng_seed))
        sqrt_dt = self.dt ** 0.5

        price_windows = []
        nxt_prices = []
        features = []
        dates = []

        for path_idx in range(self.num_paths):
            n_total = self.lookback_window + self.steps_per_path
            x = torch.zeros(n_total + 1, dtype=torch.float32)
            x[0] = self.x0_mean + self.x0_std * torch.randn(1, generator=gen, dtype=torch.float32)

            for t in range(n_total):
                x_t = x[t]
                mu_t = self._true_drift(x_t)
                sigma_t = self._true_diffusion(x_t)
                noise = torch.randn(1, generator=gen, dtype=torch.float32)
                x[t + 1] = x_t + mu_t * self.dt + sigma_t * sqrt_dt * noise

            for start in range(self.steps_per_path):
                end = start + self.lookback_window
                hist = x[start:end]
                x_t = hist[-1]

                price_windows.append(hist)
                nxt_prices.append(x[end].unsqueeze(0))
                features.append(self._make_features(hist))
                dates.append(f"synthetic_path{path_idx}_step{start}")

        self.price_window = torch.stack(price_windows)
        self.nxt_prices = torch.stack(nxt_prices)
        sample_min = self.price_window.min(dim=1, keepdim=True).values
        sample_max = self.price_window.max(dim=1, keepdim=True).values
        self.sample_min = sample_min
        self.sample_range = torch.clamp(sample_max - sample_min, min=1e-8)
        self.features = torch.stack(features)
        self.test_dates = dates

    def _true_drift(self, x_t):
        return self.kappa * (self.theta - x_t)

    def _true_diffusion(self, x_t):
        return self.sigma_base + self.sigma_scale * torch.sigmoid(x_t)

    def _make_features(self, hist):
        x_t = hist[-1]
        mean = hist.mean()
        std = hist.std(unbiased=False)
        min_v = hist.min()
        max_v = hist.max()
        return torch.stack([x_t, x_t**2, mean, std, min_v, max_v, torch.tensor(1.0)])

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return NeuralSDESample(
            price_window=self.price_window[idx],
            nxt_price=self.nxt_prices[idx],
            sample_min=self.sample_min[idx],
            sample_range=self.sample_range[idx],
            features=self.features[idx],
            fib_levels=self.features[idx],  # we can use features as fib_levels placeholder
            test_dates=self.test_dates[idx],
            issue_ids=f"synthetic_path{idx}",
        )