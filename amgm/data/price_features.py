from typing import Callable
import polars as pl
import numpy as np
from numba import njit


# Group-by operator
def group_by_issue_id(df: pl.DataFrame, func: Callable):
    return df.sort(["IssueId", "Date"]).group_by("IssueId", maintain_order=True).map_groups(func)


def add_log_value(df: pl.DataFrame, src="ClAdjUsd", dst="LogClAdjUsd") -> pl.DataFrame:
    """Calculate and add the logarithm of a given source column."""

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        return group.with_columns([pl.col(src).log().alias(dst)]).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


def add_sum(df: pl.DataFrame, srcs=["ClAdjUsd", "OpAdjUsd"], dst="SumOpClAdjUsd") -> pl.DataFrame:
    """Calculate and add the sum of the given source columns."""

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        s = sum(pl.col(src) for src in srcs)
        return group.with_columns([s.alias(dst)]).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


# Close-to-close volatility


def add_clvol(df: pl.DataFrame, src="ClAdjUsd", dst="Cl1mVol", window: int = 21) -> pl.DataFrame:
    """Calculate the rolling log return volatility for a given source column of closing prices and lookback window size."""

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        log_ret = (pl.col(src) / pl.col(src).shift(1)).log()
        log_vol = log_ret.fill_null(strategy="forward").rolling_std(window, min_samples=4)
        return group.with_columns([log_vol.alias(dst)]).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


# Hurst exponent


@njit
def hurst_exponent(prices: np.ndarray) -> float:
    if len(prices) < 20:
        # Not enough data to compute the Hurst exponent: the result will be too noisy and potentially misleading
        return np.nan
    ts = np.cumsum(prices - np.mean(prices))
    R = np.max(ts) - np.min(ts)
    S = np.std(prices)
    e = np.log(R / S) / np.log(len(prices)) if S > 0 else np.nan
    return e


def add_hurst(df: pl.DataFrame, src="ClAdjUsd", dst="Hurst1m", window: int = 21) -> pl.DataFrame:
    """Calculate and add the Hurst exponent for a given source column of prices over a rolling window of size `window`.
    This function can be quite slow."""

    def _hurst_exponent(prices: pl.Series) -> float:
        return hurst_exponent(prices.to_numpy())

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        return group.with_columns([pl.col(src).rolling_map(_hurst_exponent, window).alias(dst)]).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


# SMA Crossover


def add_sma_crossover(
    df: pl.DataFrame, src="ClAdjUsd", dst="Sma50x200", short_win: int = 50, long_win: int = 200
) -> pl.DataFrame:
    """Calculate and add the ratio of two rolling means (short and long) for a given source column of prices."""

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        return group.with_columns(
            [(pl.col(src).rolling_mean(short_win) / pl.col(src).rolling_mean(long_win)).alias(dst)]
        ).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


# Min and max 1-day return


def add_min_1d_return(df: pl.DataFrame, src="ClAdjUsd", dst="Min1dRet1m", window: int = 21) -> pl.DataFrame:
    """Calculate and add the minimum 1-day return observed over a rolling window of size `window`."""

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        ret1d = (pl.col(src) / pl.col(src).shift(1) - 1).rolling_min(window)
        return group.with_columns([ret1d.alias(dst)]).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


def add_max_1d_return(df: pl.DataFrame, src="ClAdjUsd", dst="Max1dRet1m", window: int = 21) -> pl.DataFrame:
    """Calculate and add the maximum 1-day return observed over a rolling window of size `window`."""

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        ret1d = (pl.col(src) / pl.col(src).shift(1) - 1).rolling_max(window)
        return group.with_columns([ret1d.alias(dst)])

    ret = group_by_issue_id(df, _fn)
    return ret


# Cumulative return


def add_cumulative_return(df: pl.DataFrame, src="ClAdjUsd", dst="CumRet1m", window: int = 21) -> pl.DataFrame:
    """Compute and add cumulative return over a rolling window of size `window`"""

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        return group.with_columns([(pl.col(src) / pl.col(src).shift(window) - 1).alias(dst)]).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


# Bollinger Band %B


def add_pctbb(df: pl.DataFrame, src="ClAdjUsd", dst="PctBb", window: int = 21, num_std: float = 2.0) -> pl.DataFrame:
    """Calculate and add the Bollinger Band %B for a given source column of prices."""

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        bb_mean = pl.col(src).rolling_mean(window)
        bb_std = pl.col(src).rolling_std(window)
        pct_bb = (pl.col(src) - (bb_mean - num_std * bb_std)) / (2 * num_std * bb_std)
        return group.with_columns(pct_bb.alias(dst)).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


# RSI 14


def add_rsi14(df: pl.DataFrame, src="ClAdjUsd", dst="Rsi14", window: int = 14) -> pl.DataFrame:
    """Calculate and add the 14-day Relative Strength Index (RSI) for a given source column of prices."""

    # FIXME: 14 calendar days or 14 trading days?

    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        delta = pl.col(src) - pl.col(src).shift(1)
        gain = delta.clip(lower_bound=0).rolling_mean(window)
        loss = (-delta.clip(upper_bound=0)).rolling_mean(window)
        rsi = 100 - 100 / (1 + gain / (loss + 1e-6))
        return group.with_columns([rsi.alias(dst)]).drop_nulls([dst])

    ret = group_by_issue_id(df, _fn)
    return ret


## Volume Features

# On-balance Volume

def add_obv(df:pl.DataFrame, price_col="ClAdjUsd", volume_col="VoAdj" , dst="OnBalanceVolume") -> pl.DataFrame:
    """
    Calculate On-Balance Volume (OBV).
    """
    def _fn(group: pl.DataFrame) -> pl.DataFrame:
        price = group[price_col].to_numpy()
        volume = group[volume_col].to_numpy()
        obv = [0]
        for i in range(1, len(price)):
            if price[i] > price[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif price[i] < price[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        return group.with_columns(pl.Series(dst, obv, dtype=pl.Float64))
    
    ret = group_by_issue_id(df, _fn)
    return ret
