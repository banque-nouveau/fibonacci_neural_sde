import pandas as pd
from scipy.stats import linregress
import numpy as np
from tqdm import tqdm
import torch
from numba import njit
import logging

def calculate_fibLevels(price_window):
    """Calculate Fibonacci levels using a single price window.

    Args:
        price_window (np.ndarray): Array of shape (sample_num, lookback_window).
    Returns:
        np.ndarray: Array of shape (sample_num, 7) containing Fibonacci levels for each sample.
    """
    min_price = price_window.min(axis=1, keepdims=True)
    max_price = price_window.max(axis=1, keepdims=True)
    delta = max_price - min_price

    fib_levels = np.concatenate(
        [
            min_price,
            min_price + 0.236 * delta,
            min_price + 0.382 * delta,
            min_price + 0.500 * delta,
            min_price + 0.618 * delta,
            min_price + 0.786 * delta,
            max_price,
        ],
        axis=1,
    ).astype(np.float32)

    return fib_levels, delta
    
def calculate_fibLevels_multi_window(price_windows):
    """Selects dynamic lookback sub-window based on recent volatility dominance.
    Evaluates 63d, 126d, and 252d horizons in a fully vectorized cascade.
    """
    # 1. Define sub-windows
    w_short = price_windows[:, -63:]
    w_med = price_windows[:, -126:]
    w_macro = price_windows
    
    # 2. Compute min/max for each horizon
    min_s, max_s = w_short.min(axis=1, keepdims=True), w_short.max(axis=1, keepdims=True)
    min_m, max_m = w_med.min(axis=1, keepdims=True), w_med.max(axis=1, keepdims=True)
    min_l, max_l = w_macro.min(axis=1, keepdims=True), w_macro.max(axis=1, keepdims=True)
    
    # 3. Calculate price ranges
    rng_s = max_s - min_s
    rng_m = max_m - min_m
    rng_l = max_l - min_l
    
    # Avoid divide-by-zero
    safe_rng_l = np.maximum(rng_l, 1e-8)
    
    # 4. Hierarchical conditions (evaluate short first, then medium, fallback to macro)
    # Tier 1: Short window dominates if it holds > 50% of the macro range
    cond_short = (rng_s / safe_rng_l) > 0.5
    
    # Tier 2: Medium window dominates if it holds > 50% of the macro range
    cond_med = (rng_m / safe_rng_l) > 0.5
    
    # 5. Vectorized selection using np.select
    min_price = np.select([cond_short, cond_med], [min_s, min_m], default=min_l)
    max_price = np.select([cond_short, cond_med], [max_s, max_m], default=max_l)
    
    # 6. Calculate Fibonacci levels
    delta = np.maximum(max_price - min_price, 1e-8)
    ratios = np.array([0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0], dtype=np.float32)
    
    return (min_price + delta * ratios).astype(np.float32), delta


def balanced_subsample_residuals(y, bins_fn, target_count):
    bins = bins_fn(y)
    if not isinstance(bins, dict):
        raise TypeError("bins_fn must return a dict of bin_name -> indices")
    
    print(f"Bin counts before balancing: { {bin_name: len(indices) for bin_name, indices in bins.items()} }")
    
    balanced_indices = []
    for bin, indices in bins.items():
        n_samples = len(indices)
        if n_samples == 0:
            continue
        
        # Oversample if we have too few, Undersample if we have too many
        replace = n_samples < target_count
        selected_indices = np.random.choice(indices, target_count, replace=replace)
        balanced_indices.extend(selected_indices)
    
    return y[balanced_indices]

def metric_regression_residual(y_true, y_hat, window_size_test=63, ticker=None, bins_fn=None):
    """
    Calculate the mean absolute error (MAE) of residual percentages for each bin, as well as the overall mean squared error (MSE).

    Args:
        y_true (np.ndarray): Array of true residual percentages (shape [N, 1])
        y_hat (np.ndarray): Array of predicted residual percentages (shape [N, 1])
    """

    y_true = np.array(y_true)
    y_hat = np.array(y_hat)

    bins = bins_fn(y_true)
    if not isinstance(bins, dict):
        raise TypeError("bins_fn must return a dict of bin_name -> indices")
    
    # Compute mean absolute error for each bin
    errors = y_true - y_hat
    mae_errors = []
    for bin_indices in bins.values():
        if len(bin_indices) == 0:
            mae_errors.append(np.nan)
        else:
            mae_errors.append(np.mean(np.abs(errors[bin_indices])))
    bin_names = list(bins.keys())
    
    # Compute Mean-squared error over all samples and days
    mse = np.mean(errors ** 2)
    
    return mse, mae_errors, bin_names

def metric_regression_time(y_true, y_hat):
    """
    Calculate the mean absolute error (MAE) for each label in a regression task, as well as the overall mean squared error (MSE).
    Args:
        y_true (np.ndarray): Array of true labels (shape [N, 1])
        y_hat (np.ndarray): Array of predicted labels (shape [N, 1])
    Returns:
        tuple:
            - float: Overall mean squared error (MSE).
            - list: List of MAE values for each label.
    """
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    
    unique_classes, counts = np.unique(y_true, return_counts=True)
    
    # Compute mean absolute error for each label
    errors = y_true - y_hat
    mae_errors = [np.mean(np.abs(errors[y_true == label])) for label in unique_classes]
    
    # Compute Mean-squared error over all samples
    mse = np.mean(errors ** 2)
    
    return mse, np.array(mae_errors)

def analyze_label_change_lag(y_true, y_hat, k=100):
    """
    Analyze the lag in misclassifications after a label change in a time series classification task.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_hat (np.ndarray): Array of predicted labels.
        k (int): Number of days to look ahead after a label change.

    Returns:
        tuple:
            lag_mis_pct (float): Percentage of consecutive misclassifications that occur within k days after a label change.
            avg_recovery_time (float): Average number of days it takes for the model to recover (make correct predictions) after a label change.
    """

    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    N = len(y_true)
    change_points = np.where(y_true[1:] != y_true[:-1])[0] + 1  # indices where label changes

    lag_misclassifications = []
    lag_recovery_times = []

    for idx in change_points:
        # Look at the next k days after the change
        for offset in range(k):
            j = idx + offset
            if j >= N:
                break
            if y_hat[j] != y_true[j] and y_true[idx] == y_true[j]: # Only count as misclassification if the true label has not changed again
                lag_misclassifications.append(j)
            else:
                # Recovery: first correct prediction after change
                lag_recovery_times.append(offset)
                break

    total_misclassifications = np.sum(y_hat != y_true)
    lag_mis_pct = 100 * len(lag_misclassifications) / total_misclassifications if total_misclassifications > 0 else 0
    avg_recovery_time = np.mean(lag_recovery_times)
    
    return lag_mis_pct, avg_recovery_time
    
def preprocess_dataframes(security_data):
    """
    Converts the 'Date' column in the DataFrame to datetime format of 'YYYY-MM-DD', and renaming the column to 'Date_prices'.
    """
    security_data['Date'] = pd.to_datetime(security_data['Date'])
    security_data['Date'] = security_data['Date'].dt.strftime('%Y-%m-%d')
    security_data.rename(columns={'Date': 'Date_prices'}, inplace=True)

    return security_data

def get_security_data_by_date(df, start_date="2000-01-01", end_date="2025-01-01"):
    """
    Filter the security DataFrame by a range on 'Date_prices'.
    """
    return df[(df["Date_prices"] >= start_date) & (df["Date_prices"] <= end_date)]

def balanced_subsample(x, y):
    """Create largest balanced subsample of the dataset

    Args:
        x (np.ndarray): Features of the dataset.
        y (np.ndarray): Labels of the dataset.
    Returns:
        tuple: Balanced subset of x and y
    """
    # Get the unique classes and their counts
    unique_classes, counts = np.unique(y, return_counts=True)
    
    # Find the minimum count among the classes
    min_count = np.min(counts)
    
    balanced_indices = []
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        selected_indices = np.random.choice(class_indices, min_count, replace=False)
        balanced_indices.extend(selected_indices)
    
    # Shuffle to avoid ordered samples
    np.random.shuffle(balanced_indices)

    return x[balanced_indices], y[balanced_indices]


def resample_window_size(window_size, time_bar="daily"):
    """
    Resample the window size based on the time bar frequency.
    Parameters:
    - window_size: integer representing the original window size
    - time_bar: string indicating the time bar frequency ('daily', 'weekly', 'monthly')
    Returns:
    - Resampled window size based on the time bar frequency
    """
    if time_bar == "daily":
        return window_size  # No change for daily bars
    elif time_bar == "weekly":
        return int(window_size / 5)  # Assuming 5 trading days in a week
    elif time_bar == "monthly":
        return int(window_size / 21)  # Assuming 21 trading days in a month

def resample_time_bars(df, time_bar="daily"):
    """
    Resample the DataFrame to the specified time bar frequency.
    Parameters:
    - df: DataFrame containing 'Date_prices'
    - time_bar: string indicating the time bar frequency ('daily', 'weekly', 'monthly')
    Returns:
    - DataFrame resampled to the specified time bar frequency
    """
    df = df.set_index('Date_prices')
    if time_bar == "daily":
        return df.reset_index()  # No resampling needed for daily bars
    elif time_bar == "weekly":
        print("Resampling to weekly bars")
        df = df.groupby(pd.Grouper(freq='W-FRI')).last()
        df = df.resample('W-FRI').apply(lambda x: x.iloc[-1])
        return df.reset_index()
    elif time_bar == "monthly":
        df = df.groupby(pd.Grouper(freq='ME')).last()
        df = df.resample('ME').apply(lambda x: x.iloc[-1])
        return df.reset_index()
    
def does_linear_trend_fit(close_price, test_start_index_index, window_size_train, window_size_test, time_bar, CI_threshold=1.96):
    """
    Check if a linear trend on train window fits the close price over test window for a given confidence interval.
    Parameters:
    - close_price: numpy array of daily close prices
    - test_start_index_index: starting index for the test sliding window
    - window_size_train: size of the training window
    - window_size_test: size of the test window
    - CI_threshold: threshold for confidence interval (default 1.96 for 95% CI)
    Returns:
    - coverage: percentage of test window points within the confidence interval
    """
    slope_list = []
    intercept_list = []
    
    x_train = np.arange(window_size_train)
    y_train = close_price[test_start_index_index - window_size_train:test_start_index_index]
    
    y_temp = y_train[::5]  # Resample to weekly bars
    x_temp = x_train[::5]  # Adjust x_train to match the new length
    slope, intercept, _, _, _ = linregress(x_temp, y_temp)
    slope_list.append(slope)
    intercept_list.append(intercept)
    
    y_temp = y_train[::21]  # Resample to weekly bars
    x_temp = x_train[::21]  # Adjust x_train to match the new length
    slope, intercept, _, _, _ = linregress(x_temp, y_temp)
    slope_list.append(slope)
    intercept_list.append(intercept)
    
    if time_bar == "weekly":
        y_train = y_train[::5]  # Resample to weekly bars
        x_train = x_train[::5]  # Adjust x_train to match the new length
    elif time_bar == "monthly":
        y_train = y_train[::21]  # Resample to monthly bars
        x_train = x_train[::21]  # Adjust x_train to match the new length
    
    # Fit linear regression using scipy.stats.linregress
    slope, intercept, _, _, _ = linregress(x_train, y_train)
    slope_list.append(slope)
    intercept_list.append(intercept)
    
    # Predict on test window
    x_test = np.arange(window_size_train, window_size_train + window_size_test)
    y_test = close_price[test_start_index_index:test_start_index_index + window_size_test]
    
    # Calculate predictions
    train_pred = intercept + slope * x_train
    test_pred = intercept + slope * x_test

    # Calculate confidence intervals using ±1.96 * std_err
    residuals = y_train - train_pred
    ci_width = CI_threshold * np.std(residuals)
    lower_bound = test_pred - ci_width
    upper_bound = test_pred + ci_width

    # Check coverage
    inside_ci = np.logical_and(y_test >= lower_bound, y_test <= upper_bound)
    coverage = np.mean(inside_ci)    
    
    return coverage, np.std(slope_list)/np.mean(slope_list)

def create_dataset_LT(issues_ids, start_date, end_date_train, end_date_test, window_size_train=365, window_size_test=90, time_bar="daily", file_path=None, verbose=False):
    """
    Create a dataset for linear trend prediction based on security data.
    Parameters:
    - issues_ids: list of issue IDs to process
    - start_date: start date for the data extraction
    - end_date_train: end date for the training data
    - end_date_test: end date for the test data
    - window_size_train: size of the training window (default 365)
    - window_size_test: size of the test window (default 90)
    Returns:
    - x_train: training data features
    - y_train: training data labels
    - x_test: test data features
    - y_test: test data labels
    """
    # Read the content of the file as a DataFrame
    security_data = pd.read_csv(file_path, sep='\t', dtype={'IssueId': str}) 
    security_data = preprocess_dataframes(security_data)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    CV_slope_list = []
    
    for issue_id in tqdm(issues_ids, desc="Creating dataset"):
        # Process the training data
        issueID_data = get_security_data_by_date(security_data[security_data["IssueId"] == issue_id], start_date, end_date_train)
        close_price = issueID_data["ClAdjLoc"].values
        
        n = len(close_price)
        num_windows = n - window_size_train - window_size_test + 1
        if num_windows <= 0 and verbose:
            print(f"Not enough data for issue {issue_id} to create training windows.")
        else:
            for start in range(num_windows):
                x_train.append(close_price[start:start+window_size_train].tolist())
                coverage, CV_slope = does_linear_trend_fit(close_price, start+window_size_train, window_size_train, window_size_test, time_bar)
                y_train.append([coverage if coverage == 1 else 0]) # Store coverage as label (1 for fit, 0 for no fit)
                CV_slope_list.append(CV_slope)
                
        # Now process the test data
        issueID_data = get_security_data_by_date(security_data[security_data["IssueId"] == issue_id], end_date_train, end_date_test)
        close_price = issueID_data["ClAdjLoc"].values
        
        n = len(close_price)
        num_windows = n - window_size_train - window_size_test + 1
        if num_windows <= 0 and verbose:
            print(f"Not enough data for issue {issue_id} to create test windows.")
        else: 
            for start in range(num_windows):
                x_test.append(close_price[start:start+window_size_train].tolist())
                coverage, CV_slope = does_linear_trend_fit(close_price, start+window_size_train, window_size_train, window_size_test, time_bar)
                y_test.append([coverage if coverage == 1 else 0]) # Store coverage as label (1 for fit, 0 for no fit)
                CV_slope_list.append(CV_slope)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # Ensure the shapes are correct
    if verbose:
        print(f"Train set shape: {x_train.shape}, {y_train.shape}, Test set shape: {x_test.shape}, {y_test.shape}")
    
    print(f"CV slope: {np.mean(CV_slope_list)}")
    
    return x_train, y_train, x_test, y_test

def metrics_binary_classification(y_true, y_pred, verbose=False):
    """
    Calculate accuracy, false positive rate, and false negative rate for binary classification.
    Parameters:
    - y_true: tensor of true labels
    - y_pred: tensor of predicted labels
    Returns:
    - accuracy: float, accuracy of the predictions
    - FPR: float, false positive rate
    - FNR: float, false negative rate
    - conf_matrix: tensor, confusion matrix
    """
    # Confusion matrix components
    TP = ((y_pred == 1) & (y_true == 1)).sum().item()
    TN = ((y_pred == 0) & (y_true == 0)).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item()

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    conf_matrix = torch.tensor([[TN, FP], [FN, TP]])
    
    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"False Positive Rate (FPR): {FPR:.4f}")
        print(f"False Negative Rate (FNR): {FNR:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

    return accuracy, FPR, FNR, conf_matrix

def metrics_multiclass_classification(y_true, y_pred, num_classes=3, verbose=False):
    """
    Calculate accuracy, per-class FPR, per-class FNR, and confusion matrix for multiclass classification.
    Parameters:
    - y_true: tensor of true labels
    - y_pred: tensor of predicted labels
    - num_classes: int, number of classes (default 3)
    Returns:
    - accuracy: float, overall accuracy
    - conf_matrix: tensor, confusion matrix
    """

    # Confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1

    row_sums = conf_matrix.sum(axis=1, keepdims=True)  # Sum of each row, shape (num_classes, 1)
    # Avoid division by zero by replacing zero sums with ones (will result in 0% for those rows)
    row_sums_safe = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    # Divide each element by its row sum
    CM_rate = conf_matrix / row_sums_safe * 100

    # Only average over classes present in y_true
    present_classes = (row_sums.squeeze() > 0)
    if present_classes.sum() > 0:
        accuracy = torch.diagonal(CM_rate)[present_classes].sum().item() / present_classes.sum().item()
    else:
        accuracy = float('nan')

    return accuracy, CM_rate

def cat_torch(items):
    """Concatenate torch tensors along first dim, move to CPU."""
    return torch.cat(items, dim=0).cpu()

def cat_numpy(items):
    """Concatenate NumPy arrays along first axis."""
    return np.concatenate(items, axis=0)

def merge_date_windows(items):
    """Merge collated date_windows efficiently.

    For date windows returned as Python lists, default_collate yields a
    transposed structure [T, B] (time-major). This converts each batch back
    to [B, T] and concatenates all batches on axis 0.
    """
    arrays = []
    for item in items:
        if not item:
            continue
        arr = np.asarray(item, dtype=object)
        if arr.ndim == 1:
            # Single sample case where item may already be [T].
            arr = arr.reshape(1, -1)
        else:
            # default_collate on sequence fields returns [T, B].
            arr = arr.T
        arrays.append(arr)

    if not arrays:
        return np.empty((0, 0), dtype=object)
    return np.concatenate(arrays, axis=0)

def merge_list_items(items):
    """Merge list-based batch outputs into a sample-first flat list.

    PyTorch default_collate transposes sequence fields from [B, T] to [T, B].
    For such fields (e.g. date_windows), this converts them back to [B, T]
    and then appends samples across batches.
    """
    merged_items = []
    for item in items:
        if not item:
            continue

        first_subitem = item[0]
        if isinstance(first_subitem, (list, tuple, np.ndarray)):
            # Convert collated [T, B] representation back to sample-first [B, T].
            try:
                batch_first = list(zip(*item))
                merged_items.extend([list(sample) for sample in batch_first])
            except TypeError:
                merged_items.extend(item)
        else:
            merged_items.extend(item)

    return merged_items

def merge_results(results, torch_fn, numpy_fn):
    """
    Merge a list of dictionaries with batch outputs from Trainer.predict().

    Args:
        results (list[dict]): output from Trainer.predict() where each item is a dict
        torch_fn (callable): function to concat tensors (e.g., cat_torch)
        numpy_fn (callable): function to concat arrays (e.g., cat_numpy)

    Returns:
        dict: merged outputs with same keys as input dicts, concatenated across batches
    """
    merged = {}
    # Collect list of all keys from first dictionary (assumes all dicts have same keys)
    keys = results[0].keys() if results else []
    
    for key in keys:
        # Gather all items for this key across results
        items = [r[key] for r in results]

        # # date_windows are only used in single-sample plotting flows.
        # # Avoid expensive list/object merging and keep the first batch as-is.
        if key == "date_windows":
            merged[key] = items[0]
            continue

        # Determine concat method by inspecting first item
        first_item = items[0]
        if torch.is_tensor(first_item):
            merged[key] = torch_fn(items)
        elif isinstance(first_item, list):
            merged[key] = merge_list_items(items)
        else:
            try:
                merged[key] = numpy_fn(items)
            except Exception as exc:
                # Build detailed diagnostics to identify the exact bad key/item.
                item_debug = []
                for idx, item in enumerate(items):
                    shape = getattr(item, "shape", None)
                    dtype = getattr(item, "dtype", None)
                    ndim = getattr(item, "ndim", None)
                    preview = ""

                    # Helpful for object arrays or short 1D metadata arrays such as dates.
                    if ndim == 1 and len(item) <= 5:
                        try:
                            preview = f", sample={item.tolist()}"
                        except Exception:
                            preview = ""

                    item_debug.append(
                        f"idx={idx}, type={type(item).__name__}, shape={shape}, dtype={dtype}{preview}"
                    )

                raise ValueError(
                    "merge_results failed while concatenating key "
                    f"'{key}' with numpy_fn={getattr(numpy_fn, '__name__', str(numpy_fn))}. "
                    "Per-item diagnostics: "
                    + " | ".join(item_debug)
                ) from exc
    
    return merged

def unpack_predict_results(results):
    """
    Unpack results from Trainer.predict() into separate tensors.

    Args:
        results (list[dict]): output from Trainer.predict()

    Returns:
        tuple: unpacked tensors for y_pred, targets, features, and any additional fields
    """
    # Access merged tensors/numpy arrays by key:
    y_pred = results['y_pred']
    p_pred = results['p_pred']
    targets = results['targets']
    features = results['features']
    issue_ids = results.get('issue_ids')  # use .get() if keys might be missing in some cases
    test_dates = results.get('test_dates')
    slopes = results.get('slopes')
    deviations = results.get('deviations')
    test_prices = results.get('test_prices')
    ci_widths = results.get('ci_widths')
    ci_lowers = results.get('ci_lowers')
    ci_uppers = results.get('ci_uppers')
    margins = results.get('margins')
    price_windows = results.get('price_windows')

    return (y_pred, p_pred, targets, features, issue_ids, test_dates, slopes, deviations, test_prices, ci_widths, ci_lowers, ci_uppers, margins, price_windows)

@njit
def fast_linreg_numba(x, y):
    """Fast linear regression using Numba for performance."""
    x_mean = 0.0
    y_mean = 0.0
    n = len(x)
    for i in range(n):
        x_mean += x[i]
        y_mean += y[i]
    x_mean /= n
    y_mean /= n

    num = 0.0
    den = 0.0
    for i in range(n):
        dx = x[i] - x_mean
        dy = y[i] - y_mean
        num += dx * dy
        den += dx * dx

    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept

def _safe_corr(x, y, eps=1e-12):
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.pow(2).sum() + eps) * (y_centered.pow(2).sum() + eps))
    return (x_centered * y_centered).sum() / denom


def evaluate_synthetic_recovery(predictions, dset_cfg, run_cfg):
    """ Evaluate recovery of the following synthetic SDE parameters from predictions:
    - drift: kappa * (theta - x_t)
    - diffusion: sigma_base + sigma_scale * sigmoid(x_t)
    """
    if not predictions:
        return None

    x_t = torch.cat([batch["x_t"] for batch in predictions], dim=0).view(-1)
    drift_pred = torch.cat([batch["drift"] for batch in predictions], dim=0).view(-1)
    diffusion_pred = torch.cat([batch["diffusion"] for batch in predictions], dim=0).view(-1)

    kappa = float(dset_cfg["kappa"])
    theta = float(dset_cfg["theta"])
    sigma_base = float(dset_cfg["sigma_base"])
    sigma_scale = float(dset_cfg["sigma_scale"])

    drift_true = kappa * (theta - x_t)
    diffusion_true = sigma_base + sigma_scale * torch.sigmoid(x_t)

    metrics = {
        "drift_mae": torch.mean(torch.abs(drift_pred - drift_true)).item(),
        "drift_corr": _safe_corr(drift_true, drift_pred).item(),
        "diffusion_mae": torch.mean(torch.abs(diffusion_pred - diffusion_true)).item(),
        "diffusion_corr": _safe_corr(diffusion_true, diffusion_pred).item(),
    }
    
    min_drift_corr = float(dset_cfg.get("min_drift_corr", -1.0))
    min_diff_corr = float(dset_cfg.get("min_diffusion_corr", -1.0))
    fail_on_recovery = bool(run_cfg.get("fail_on_bad_recovery", False))
    if fail_on_recovery:
        if metrics["drift_corr"] < min_drift_corr or metrics["diffusion_corr"] < min_diff_corr:
            print(
                "Synthetic recovery thresholds not met: "
                f"drift_corr={metrics['drift_corr']:.4f}, diffusion_corr={metrics['diffusion_corr']:.4f}."
            )
            
    return metrics

def evaluate_residual_calibration(predictions, dset_cfg):
    """ Compute standardized residuals:
    z_t = (x_{t+1} - x_t - drift_t * dt) / (diffusion_t * sqrt(dt))
    
    where:
        x_t: observed price at time t
        x_{t+1}: observed price at time t+1
        drift_t: predicted drift at time t
        diffusion_t: predicted diffusion at time t
        dt: time step size
    """
    if not predictions:
        return None

    dt = float(dset_cfg.get("dt", 1.0))
    sqrt_dt = dt ** 0.5
    eps = 1e-8

    x_t = torch.cat([batch["x_t"] for batch in predictions], dim=0).view(-1)
    x_tp1 = torch.cat([batch["x_tp1"] for batch in predictions], dim=0).view(-1)
    drift = torch.cat([batch["drift"] for batch in predictions], dim=0).view(-1)
    diffusion = torch.cat([batch["diffusion"] for batch in predictions], dim=0).view(-1)
    
    print(f"Evaluating residual calibration on {x_t.numel()} samples.")
    print(f"Shape of x_t: {x_t.shape}, x_tp1: {x_tp1.shape}, drift: {drift.shape}, diffusion: {diffusion.shape}")

    z = ((x_tp1 - x_t) - drift * dt) / (diffusion * sqrt_dt + eps)
    n = int(z.numel())

    metrics = {
        "n_residuals": n,
        "z_mean": z.mean().item(),
        "z_std": z.std(unbiased=False).item(),
        "coverage_1sigma": (z.abs() <= 1.0).float().mean().item(),
        "coverage_2sigma": (z.abs() <= 2.0).float().mean().item(),
        "coverage_3sigma": (z.abs() <= 3.0).float().mean().item(),
        "coverage_1sigma_target": 0.682689,
        "coverage_2sigma_target": 0.954500,
        "coverage_3sigma_target": 0.997300,
    }

    if n > 1:
        z0 = z[:-1]
        z1 = z[1:]
        z0c = z0 - z0.mean()
        z1c = z1 - z1.mean()
        denom = torch.sqrt((z0c.pow(2).sum() + eps) * (z1c.pow(2).sum() + eps))
        metrics["z_lag1_autocorr"] = ((z0c * z1c).sum() / denom).item()

        zc = z - z.mean()
        zvar = zc.pow(2).mean()
        zstd = torch.sqrt(zvar + eps)
        metrics["z_skew"] = (zc.pow(3).mean() / (zstd.pow(3) + eps)).item()
        metrics["z_excess_kurtosis"] = (zc.pow(4).mean() / (zvar.pow(2) + eps) - 3.0).item()
    else:
        metrics["z_lag1_autocorr"] = None
        metrics["z_skew"] = None
        metrics["z_excess_kurtosis"] = None

    return metrics

def print_dict(d):
    for key, value in d.items():
        if value is None:
            print(f"  - {key}: n/a")
        elif isinstance(value, float):
            print(f"  - {key}: {value:.6f}")
        else:
            print(f"  - {key}: {value}")
            
def debug_value_info(name_str, value):
    """ Prints the type, shape, and value of a variable for debugging purposes.
    
    For example:
        debug_value_info("my_array", my_array)
    """
    value_type = type(value).__name__
    shape = getattr(value, "shape", None)
    logging.info("%s: type=%s, shape=%s, value=%s", name_str, value_type, shape, value)
    
def sanity_check_path(synthetic_path, max_mult: float = 4.0, min_mult: float = 0.05):
    """ Check if the synthetic path is valid.
    """
    S0 = synthetic_path[0]
    
    # 1. Non-positivity check (prices must remain strictly positive)
    if np.any(synthetic_path <= 0):
        return False
        
    # 2. Explosive upper-bound check (e.g., 4x in 100 days)
    if np.max(synthetic_path) / S0 > max_mult:
        return False
        
    # 3. Collapse lower-bound check (e.g., dropping below 95% of initial value)
    if np.min(synthetic_path) / S0 < min_mult:
        return False
        
    # 4. Single-day unrealism check (daily return > 100% or drop > 90% without jump diffusion)
    returns = np.abs(np.diff(synthetic_path) / synthetic_path[:-1])
    if np.max(returns) > 1.0:  # Single-day doubling
        return False

    return True