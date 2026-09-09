import datetime as dt
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from amgm import config as amgm_config
from amgm.data.base import FeatureNormalizer
from amgm.data.linear_trend import LinearTrendSample
from amgm.models.linear_trend.runner import LinearTrendRunner
from amgm.utils import common
    
TaskName = Literal["classification", "time", "residual"]

_TASK_LOG_DIR = {
    "classification": "train_classification",
    "time": "train_regression_time",
    "residual": "train_regression_residual",
}

_CLASS_NAMES = ["Down", "NoTouch", "Up"]

def _ensure_trainer_cfg_on_path() -> None:
    """Required for unpickling legacy checkpoints that reference trainer_cfg functions."""
    experiments_dir = Path(__file__).resolve().parents[3] / "experiments" / "linear_trend"
    experiments_dir_str = str(experiments_dir)
    if experiments_dir.exists() and experiments_dir_str not in sys.path:
        sys.path.insert(0, experiments_dir_str)


def _parse_model_name(model_name: str) -> tuple[str, str, int, int, str]:
    if model_name.endswith("_US_Stocks"):
        base = model_name[: -len("_US_Stocks")]
        training_data_type = "US_Stocks"
    elif model_name.endswith("_RX1"):
        base = model_name[: -len("_RX1")]
        training_data_type = "RX1"
    else:
        raise ValueError(
            "model_name must end with '_US_Stocks' or '_RX1'. "
            "Example: 'daily_378_63_US_Stocks'."
        )

    parts = base.split("_")
    if len(parts) != 3:
        raise ValueError(
            "model_name must follow '<timebar>_<window_train>_<window_test>_<training_data_type>'."
        )

    time_bar = parts[0]
    window_size_train = int(parts[1])
    window_size_test = int(parts[2])
    return base, time_bar, window_size_train, window_size_test, training_data_type
        
class _SingleWindowDataset(Dataset):
    def __init__(
        self,
        ticker_df: pd.DataFrame,
        ticker: str,
        test_date: dt.date,
        window_size_train: int,
        window_size_test: int,
        task_type: Literal["classification", "time", "residual", "regression"],
        time_bar: str = "daily",
        normalization: Optional[dict[str, str]] = None,
    ) -> None:
        required_cols = {"Date", "ClAdjLoc"}
        missing_cols = required_cols.difference(set(ticker_df.columns))
        if missing_cols:
            raise ValueError(f"Input dataframe is missing required columns: {sorted(missing_cols)}")

        df = ticker_df.copy().sort_values("Date").reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df["ClAdjLoc"] = pd.to_numeric(df["ClAdjLoc"], errors="coerce")
        df = df.dropna(subset=["Date", "ClAdjLoc"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("Input dataframe has no valid rows after parsing Date and ClAdjLoc.")

        trading_days = pd.DatetimeIndex(df["Date"])
        test_date_ts = pd.Timestamp(test_date)
        idx = trading_days.get_indexer(pd.DatetimeIndex([test_date_ts]), method="nearest")[0]

        start_idx = idx - window_size_train + 1
        if start_idx < 0:
            raise ValueError(
                "Not enough historical rows before test_date for the requested window_size_train. "
                f"Need at least {window_size_train} rows up to the selected date."
            )

        end_idx = min(len(trading_days) - 1, idx + window_size_test)
        window_dates = trading_days[start_idx : end_idx + 1]
        mask = df["Date"].isin(window_dates)

        raw_features = df["ClAdjLoc"].iloc[start_idx : idx + 1].to_numpy(dtype=np.float32)
        x_min, x_max = raw_features.min(), raw_features.max()
        normalized_features = (raw_features - x_min) / (x_max - x_min + 1e-6)

        xs = np.ascontiguousarray(np.array([normalized_features[:, None]], dtype=np.float32))
        iids = np.array([ticker], dtype=np.str_)
        test_dates = np.array([trading_days[idx].strftime("%Y-%m-%d")], dtype=np.str_)

        has_full_test_window = (idx + window_size_test) < len(trading_days)
        if has_full_test_window:
            output_dic = self._compute_trend_metrics(
                values=df["ClAdjLoc"].to_numpy(dtype=np.float32),
                test_date_index=idx,
                train_win_size=window_size_train,
                test_win_size=window_size_test,
                time_bar=time_bar,
                normalization=normalization,
            )
            target_key = {
                "classification": "label_class",
                "time": "label_time",
                "residual": "label_residual",
            }.get(task_type, "label_residual")
            ys = np.array([[output_dic[target_key]]], dtype=np.float32)
            slopes = np.array([output_dic["slope"]], dtype=np.float32)
            deviations = np.array([output_dic["deviation_test_window"]], dtype=np.float32)
            ci_widths = np.array([output_dic["ci_width_norm"]], dtype=np.float32)
            ci_lowers = np.array([output_dic["lower_bound"]], dtype=np.float32)
            ci_uppers = np.array([output_dic["upper_bound"]], dtype=np.float32)
            margins = np.array([output_dic["margin"]], dtype=np.float32)
            test_prices = np.array([output_dic["test_price"]], dtype=np.float32)
            price_windows = np.array([output_dic["price_window"]], dtype=np.float32)
        else:
            ys = np.ascontiguousarray(np.array([[-100.0]], dtype=np.float32))
            deviations = np.array([-100.0] * window_size_test, dtype=np.float32)
            slopes = np.array([-100.0], dtype=np.float32)
            ci_widths = np.array([-100.0], dtype=np.float32)
            ci_lowers = np.array([np.array([-100.0] * window_size_test, dtype=np.float32)], dtype=np.float32)
            ci_uppers = np.array([np.array([-100.0] * window_size_test, dtype=np.float32)], dtype=np.float32)
            margins = np.array([-100.0], dtype=np.float32)
            test_prices = np.array([df["ClAdjLoc"].iloc[idx]], dtype=np.float32)
            price_windows = np.array([df.loc[mask, "ClAdjLoc"].to_numpy(dtype=np.float32)], dtype=np.float32)

        self.features = xs
        self.targets = ys.astype(int) if task_type == "classification" else ys.astype(np.float32)
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

    @staticmethod
    def _get_label_key(task_type: str) -> str:
        return {
            "classification": "label_class",
            "time": "label_time",
            "residual": "label_residual",
        }.get(task_type, "label_residual")

    @staticmethod
    def _compute_trend_metrics(
        values: np.ndarray,
        test_date_index: int,
        train_win_size: int,
        test_win_size: int,
        time_bar: str,
        normalization: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        values = np.asarray(values, dtype=np.float32)
        x_index = np.arange(train_win_size + test_win_size, dtype=np.float32)
        x_train = x_index[:train_win_size]
        y_train = values[test_date_index - train_win_size + 1 : test_date_index + 1]

        d = dict(daily=1, weekly=5, monthly=21)[time_bar]
        y_train = y_train[::d]
        x_train = x_train[::d]

        slope, intercept = common.fast_linreg_numba(x_train, y_train)

        norm_cfg = normalization or {}
        scale_fn = getattr(FeatureNormalizer, norm_cfg.get("ClAdjLoc", "sample_minmax"), None)
        if scale_fn is None:
            scale_fn = FeatureNormalizer.sample_minmax

        y_train_new = scale_fn(y_train[:, np.newaxis].T).flatten()
        slope_norm, _ = common.fast_linreg_numba(x_train, y_train_new)
        test_price = float(y_train[-1])

        x_test = x_index[-test_win_size:]
        y_test = values[test_date_index + 1 : test_date_index + test_win_size + 1]

        train_pred = intercept + slope * x_train
        test_pred = intercept + slope * x_test

        residuals = y_train - train_pred
        ci_width = 1.96 * np.std(residuals)
        lower_bound = test_pred - ci_width
        upper_bound = test_pred + ci_width
        ci_width_norm = ci_width / (max(y_train) - min(y_train))

        above = y_test > upper_bound
        below = y_test < lower_bound
        inside_ci = ~above & ~below

        any_above = above.any()
        any_below = below.any()
        if inside_ci.all():
            label_class = 1
            label_time = test_win_size
        elif any_above and any_below:
            time_upper = np.where(above)[0][0]
            time_lower = np.where(below)[0][0]
            if time_upper < time_lower:
                label_class = 2
                label_time = time_upper
            else:
                label_class = 0
                label_time = time_lower
        elif any_above:
            label_class = 2
            label_time = np.where(above)[0][0]
        else:
            label_class = 0
            label_time = np.where(below)[0][0]

        deviation_test_window = np.zeros_like(y_test, dtype=np.float32)
        mask = y_test >= upper_bound
        deviation_test_window[mask] = 100 * (y_test[mask] - upper_bound[mask]) / upper_bound[mask]

        mask = y_test <= lower_bound
        deviation_test_window[mask] = 100 * (y_test[mask] - lower_bound[mask]) / lower_bound[mask]

        if test_win_size == 63:
            deviation_test_window = np.clip(deviation_test_window, -50, 50)
        elif test_win_size == 5:
            deviation_test_window = np.clip(deviation_test_window, -15, 15)

        max_deviation = float(deviation_test_window[np.argmax(np.abs(deviation_test_window))])
        price_window = values[test_date_index - train_win_size + 1 : test_date_index + test_win_size + 1]

        initial_price = float(y_test[0])
        if initial_price < float(lower_bound[0]) or initial_price > float(upper_bound[0]):
            margin = 0.0
        else:
            ci_range = float(upper_bound[0] - lower_bound[0])
            if ci_range > 0:
                position = (initial_price - float(lower_bound[0])) / ci_range
                margin = float(np.clip(1.0 - 2.0 * abs(position - 0.5), 0.0, 1.0))
            else:
                margin = 0.0

        return {
            "label_class": float(label_class),
            "label_time": float((test_win_size - label_time) * (-1 if label_class == 0 else 1)),
            "label_residual": max_deviation,
            "slope": float(slope_norm),
            "deviation_test_window": deviation_test_window,
            "test_price": test_price,
            "ci_width_norm": float(ci_width_norm),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "margin": margin,
            "price_window": price_window,
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> LinearTrendSample:
        return LinearTrendSample(
            features=self.features[idx],
            targets=self.targets[idx],
            issue_ids=self.issue_ids[idx],
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


class LTModel:
    """Simple interface for loading linear-trend ensembles and predicting one window.

    Example:
        model = LTModel(model_name="daily_378_63_US_Stocks")
        out = model.predict(df=ticker_df, ticker="NVDA", test_date="2024-06-01")
    
    Available model_name options:
        - daily_378_63_US_Stocks
        - daily_21_5_US_Stocks
        - daily_21_5_RX1
    """

    def __init__(self, model_name: str, tasks: tuple[TaskName, ...] = ("classification", "time", "residual")):
        _ensure_trainer_cfg_on_path()

        self.model_name = model_name
        self.model_name_option, self.time_bar, self.window_size_train, self.window_size_test, self.training_data_type = _parse_model_name(
            model_name
        )
        self.tasks = tasks
        self.wdir = amgm_config.work_dir("linear_trend")
        self.models = {task: self._load_models(task) for task in tasks}

    @classmethod
    def from_options(cls, model_name_option: str, training_data_type: Literal["US_Stocks", "RX1"]) -> "LTModel":
        return cls(model_name=f"{model_name_option}_{training_data_type}")

    def _load_models(self, task: TaskName) -> list[LinearTrendRunner]:
        checkpoint_dir = self.wdir / "logs" / _TASK_LOG_DIR[task] / self.model_name / "checkpoints"
        checkpoint_files = sorted(checkpoint_dir.glob("*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found under: {checkpoint_dir}")

        return [LinearTrendRunner.load_from_checkpoint(ckpt_path, weights_only=False) for ckpt_path in checkpoint_files]

    @staticmethod
    def _trainer() -> Trainer:
        return Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    @staticmethod
    def _normalize_date(test_date: Optional[Any], ticker_df: pd.DataFrame) -> dt.date:
        if test_date is None:
            parsed_dates = pd.to_datetime(ticker_df["Date"], errors="coerce").dropna()
            if parsed_dates.empty:
                raise ValueError("Cannot infer test_date because dataframe has no valid Date values.")
            return parsed_dates.max().date()

        ts = pd.Timestamp(test_date)
        if pd.isna(ts):
            raise ValueError(f"Invalid test_date: {test_date}")
        return ts.date()

    def _predict_task(
        self,
        task: TaskName,
        ticker_df: pd.DataFrame,
        ticker: str,
        test_date: dt.date,
    ) -> dict[str, Any]:
        dataset = _SingleWindowDataset(
            ticker_df=ticker_df,
            ticker=ticker,
            test_date=test_date,
            window_size_train=self.window_size_train,
            window_size_test=self.window_size_test,
            task_type=task,
            time_bar=self.time_bar,
            normalization={"ClAdjLoc": "sample_minmax"},
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        first_outputs = None
        all_preds = []
        trainer = self._trainer()
        for mdl in self.models[task]:
            results = trainer.predict(mdl, dataloaders=loader)
            
            # Keep merge_results only once for metadata extraction.
            if first_outputs is None:
                first_outputs = {
                    "target": results[0]["targets"].item(),
                    "issue_ids": results[0]["issue_ids"][0],
                    "test_date_used": results[0]["test_dates"][0],
                    "price_windows": results[0]["price_windows"][0],
                    "deviations": results[0]["deviations"][0],
                    "slopes": results[0]["slopes"][0],
                    "difficulty_score": results[0]["margins"][0],
                }

            pred_key = "p_pred" if task == "classification" else "y_pred"
            all_preds.append(results[0][pred_key].detach().cpu().unsqueeze(0))  # Shape: (1, num_classes) or (1, 1)
        
        pred_total = torch.cat(all_preds, dim=1)
        
        outputs = first_outputs
        
        if task == "classification":
            p_hat_mean = pred_total.mean(dim=1)
            y_hat = torch.argmax(p_hat_mean, dim=1)
            probs = p_hat_mean[0].detach().cpu().numpy()
            pred_class = int(y_hat[0].item())

            outputs.update(
                {
                    "predicted_class": pred_class,
                    "predicted_label": _CLASS_NAMES[pred_class],
                    "probabilities": {
                        "Down": float(probs[0]),
                        "NoTouch": float(probs[1]),
                        "Up": float(probs[2]),
                    },
                    "ensemble_probabilities": pred_total[0].detach().cpu().numpy(),
                }
            )
        else:
            y_hat_total = pred_total.squeeze(-1)
            y_hat = y_hat_total.mean(dim=1)
            y_hat_std = y_hat_total.std(dim=1)
            outputs.pop("difficulty_score")     # Remove difficulty_score for regression tasks since it's not applicable
            outputs.update(
                {
                    "predicted_value": float(y_hat[0].item()),
                    "predicted_std": float(y_hat_std[0].item()),
                    "ensemble_predictions": y_hat_total[0].detach().cpu().numpy(),
                }
            )

        return outputs

    def predict(self, df: pd.DataFrame, ticker: str, test_date: Optional[Any] = None) -> dict[str, Any]:
        """Predict one window for all three tasks: classification, time regression, and residual(%) regression.

        Args:
            df: DataFrame with columns Date and ClAdjLoc.
            ticker: Asset identifier to carry through outputs.
            test_date: Any pandas-compatible date; nearest available trading day is used.

        Returns:
            Dict with task keys: classification, time, residual.
            
            Following keys will be set to -100.0 if test window is not fully available: target, deviations, slope, difficulty_score.
            
        | Task | Key | Type |
        | :--- | :--- | :--- |
        | **Shared** | issue_id | str | Set to the provided ticker.
        | **Shared** | test_date_used | str | Set to the actual test date used (nearest trading day).
        | **Shared** | price_window | np.ndarray | Set to the full price window used for the prediction (train + test).
        | **Shared** | target | float | Set to the true label value.
        | **Shared** | deviations | np.ndarray | Set to the true deviations (%) for the test window.
        | **Shared** | slope | float | Set to the true slope of the linear trend.
        | **Classification** | predicted_class | int | One of 0, 1, or 2 corresponding to "Down", "NoTouch", or "Up".
        | **Classification** | predicted_label | str | One of "Down", "NoTouch", or "Up".
        | **Classification** | probabilities | dict[str, float] | Keys: "Down", "NoTouch", "Up"; values are probabilities for each class.
        | **Classification** | ensemble_probabilities | np.ndarray | An array of shape (num_models=10, num_classes=3)
        | **Classification** | difficulty_score | float | Between 0.0 and 1.0; higher values indicate a more difficult prediction (closer to the center of the confidence interval).
        | **Time/Residual** | predicted_value | float |
        | **Time/Residual** | predicted_std | float |
        | **Time/Residual** | ensemble_predictions | np.ndarray | An array of shape (num_models=10, 1) for time/residual regression.
        """
        normalized_test_date = self._normalize_date(test_date, df)
        ticker = (ticker or "").strip().upper()
        if not ticker:
            raise ValueError("ticker cannot be empty.")

        return {
            task: self._predict_task(task, ticker_df=df, ticker=ticker, test_date=normalized_test_date)
            for task in self.tasks
        }
