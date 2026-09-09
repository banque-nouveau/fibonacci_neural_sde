import argparse
import importlib
import multiprocessing
import warnings
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import cast

import numpy as np
import polars as pl
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import logging

from amgm import config as amgm_config
from amgm.utils import common
from amgm.data.base import BaseAMData
from amgm.data.linear_trend import LinearTrendDataModule, LinearTrendDataset
from amgm.models.linear_trend.runner import LinearTrendRunner

# For parallel run is necessary otherwise, torch will overload each vCPU 
torch.set_num_threads(1)          # limit PyTorch to 1 thread for intra-op parallelism
torch.set_num_interop_threads(1)  # limit inter-op parallelism to 1 thread


def _get_synthetic_csv_path(neural_SDE_model_type: str) -> Path:
    """Resolve synthetic CSV path from known locations."""
    candidates = [
        amgm_config.dataset_root / "Synthetic" / f"synthetic_rollout_paths_{neural_SDE_model_type}.csv",
        Path(f"/home/alireza_javid_seb_se/asset_management/asset_management/Data/Synthetic/synthetic_rollout_paths_{neural_SDE_model_type}.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Synthetic CSV file not found. Checked: " + ", ".join(str(p) for p in candidates)
    )


def _build_augmented_train_dataset(dmod: LinearTrendDataModule, neural_SDE_model_type: str) -> LinearTrendDataset:
    """Create a train-only dataset by appending synthetic rows to local training data."""
    train_kwargs = {k: v for k, v in dmod.train_cfg.items() if k != "_target_"}
    issue_ids = cast(list[str], train_kwargs["issue_ids"])
    start_date = cast(str, train_kwargs["start_date"])
    end_date_train = cast(str, train_kwargs["end_date_train"])
    end_date_test = cast(str, train_kwargs["end_date_test"])
    window_size_train = cast(int, train_kwargs["window_size_train"])
    window_size_test = cast(int, train_kwargs["window_size_test"])
    task_type = cast(str, train_kwargs["task_type"])
    label_variable = cast(str, train_kwargs["label_variable"])
    time_bar = cast(str, train_kwargs["time_bar"])
    trend_value_col = cast(str, train_kwargs["trend_value_col"])
    feature_names = cast(list[str], train_kwargs["feature_names"])
    normalization = cast(dict[str, str] | None, train_kwargs.get("normalization"))
    balancing = train_kwargs.get("balancing")
    training_data_type = cast(str, train_kwargs["training_data_type"])
    data_source = cast(str, train_kwargs["data_source"])
    dset_path = cast(Path | None, train_kwargs.get("dset_path"))
    split = cast(str, train_kwargs["split"])

    base_secs = BaseAMData.load_data(
        training_data_type=training_data_type,
        issue_ids=issue_ids,
        start_date=start_date,
        end_date=end_date_train,
        data_source=data_source,
        dset_path=dset_path,
        uploaded_df=None,
    )

    synth_path = _get_synthetic_csv_path(neural_SDE_model_type)
    # Force IssueId to string because mixed numeric/alphanumeric IDs (e.g. 02399401C) break integer inference.
    synth = pl.read_csv(
        synth_path,
        schema_overrides={"IssueId": pl.Utf8, "Date": pl.Utf8, "ClAdjLoc": pl.Float64},
        infer_schema_length=10000,
    )

    required_cols = ["IssueId", "Date", "ClAdjLoc"]
    missing = [c for c in required_cols if c not in synth.columns]
    if missing:
        raise ValueError(f"Synthetic CSV is missing required columns: {missing}")

    synth = synth.select(required_cols)
    synth = synth.with_columns(
        [
            pl.col("IssueId").str.strip_chars().cast(pl.Utf8),
            pl.col("Date").cast(pl.Date, strict=False),
            pl.col("ClAdjLoc").cast(pl.Float64),
        ]
    )
    synth = synth.drop_nulls(["IssueId", "Date", "ClAdjLoc"])

    d1 = datetime.strptime(start_date, "%Y-%m-%d").date()
    d2 = datetime.strptime(end_date_train, "%Y-%m-%d").date()
    synth = synth.filter((pl.col("Date") >= d1) & (pl.col("Date") <= d2))

    base_secs = base_secs.select(required_cols)
    combined = pl.concat([base_secs, synth], how="vertical_relaxed")
    # combined = combined.unique(subset=["IssueId", "Date"], keep="last")

    synth_issue_ids = synth["IssueId"].drop_nulls().unique().to_list()
    issue_ids_augmented = sorted(set(issue_ids) | set(synth_issue_ids))

    print(
        "Train augmentation stats - "
        f"base_rows={base_secs.height}, synth_rows={synth.height}, combined_rows={combined.height}, "
        f"train_issue_ids_before={len(issue_ids)}, train_issue_ids_after={len(issue_ids_augmented)}"
    )

    return LinearTrendDataset(
        split=split,
        issue_ids=issue_ids_augmented,
        start_date=start_date,
        end_date_train=end_date_train,
        end_date_test=end_date_test,
        window_size_train=window_size_train,
        window_size_test=window_size_test,
        task_type=task_type,
        label_variable=label_variable,
        time_bar=time_bar,
        trend_value_col=trend_value_col,
        feature_names=feature_names,
        normalization=cast(dict[str, str], normalization),
        balancing=balancing,
        training_data_type=training_data_type,
        data_source=data_source,
        dset_path=dset_path,
        uploaded_df=combined,
    )


def _balance_smallest_with_synthetic_share(
    dataset: LinearTrendDataset,
    synthetic_share_of_total: float,
    target_total_samples: int = 60000,
    synthetic_tag: str = "synthetic",
) -> LinearTrendDataset:
    """Build a fixed-size class-balanced subset with target synthetic share per class."""
    share = float(synthetic_share_of_total)
    if share > 1.0:
        share = share / 100.0
    if not (0.0 <= share <= 1.0):
        raise ValueError(
            f"synthetic_share_of_total must be in [0, 1] or [0, 100], got {synthetic_share_of_total}"
        )

    if len(dataset) == 0:
        return dataset

    y = dataset.targets
    y_flat = y.flatten()
    unique_classes, counts = np.unique(y_flat, return_counts=True)
    if len(counts) == 0:
        print("Synthetic balancing skipped: no classes found in train dataset.")
        return dataset

    n_classes = int(len(unique_classes))

    requested_n_per_class = target_total_samples // n_classes
    target_synth = int(round(requested_n_per_class * share))
    target_real = requested_n_per_class - target_synth

    issue_ids_str = dataset.issue_ids.astype(str)
    synthetic_mask = np.char.find(np.char.lower(issue_ids_str), synthetic_tag.lower()) >= 0
    class_caps = {}
    for class_label in unique_classes:
        class_indices = np.where(y_flat == class_label)[0]
        synth_class_indices = class_indices[synthetic_mask[class_indices]]
        real_class_indices = class_indices[~synthetic_mask[class_indices]]
        s_i = int(len(synth_class_indices))
        r_i = int(len(real_class_indices))
        t_i = int(len(class_indices))

        if share == 0.0:
            max_n_i = r_i
        elif share == 1.0:
            max_n_i = s_i
        else:
            max_n_i = int(np.floor(min(s_i / share, r_i / (1.0 - share))))

        feasible_for_request = (s_i >= target_synth) and (r_i >= target_real)
        class_caps[int(class_label)] = {
            "total": t_i,
            "synthetic": s_i,
            "real": r_i,
            "current_share": (s_i / t_i) if t_i > 0 else float("nan"),
            "max_n_per_class": max_n_i,
            "requested_n_per_class": requested_n_per_class,
            "feasible_for_request": feasible_for_request,
        }

    class_caps_lines = []
    for class_id in sorted(class_caps):
        cap = class_caps[class_id]
        class_caps_lines.append(
            "  "
            f"class {class_id}: total={cap['total']}, synth={cap['synthetic']}, "
            f"real={cap['real']}, current_share={cap['current_share']:.4f}, "
            f"max_n_per_class={cap['max_n_per_class']}, "
            f"requested_n_per_class={cap['requested_n_per_class']}, "
            f"feasible_for_request={cap['feasible_for_request']}"
        )

    print(
        "Synthetic class-balanced planning:\n"
        f"  requested_share={share:.4f}\n"
        f"  target_total_samples={target_total_samples}\n"
        f"  requested_n_per_class={requested_n_per_class}\n"
        f"  target_synth_per_class={target_synth}\n"
        f"  target_real_per_class={target_real}\n"
        "  class_caps:\n"
        + "\n".join(class_caps_lines)
    )
        
    balanced_indices: list[int] = []
    class_stats = {}
    for class_label in unique_classes:
        class_indices = np.where(y_flat == class_label)[0]
        synth_class_indices = class_indices[synthetic_mask[class_indices]]
        real_class_indices = class_indices[~synthetic_mask[class_indices]]
        assert len(synth_class_indices) >= target_synth
        assert len(real_class_indices) >= target_real

        selected_synth = (
            np.random.choice(synth_class_indices, target_synth, replace=False)
            if target_synth > 0
            else np.array([], dtype=int)
        )
        selected_real = (
            np.random.choice(real_class_indices, target_real, replace=False)
            if target_real > 0
            else np.array([], dtype=int)
        )

        selected_indices = np.concatenate([selected_synth, selected_real])
        np.random.shuffle(selected_indices)
        balanced_indices.extend(selected_indices.tolist())

        class_stats[int(class_label)] = {
            "available_total": int(class_indices.size),
            "available_synth": int(synth_class_indices.size),
            "available_real": int(real_class_indices.size),
            "kept_synth": int(target_synth),
            "kept_real": int(target_real),
            "kept_total": int(requested_n_per_class),
            "kept_share": float(target_synth / requested_n_per_class) if requested_n_per_class > 0 else float("nan"),
        }

    kept_indices = np.array(balanced_indices, dtype=int)
    np.random.shuffle(kept_indices)

    row_fields = [
        "features",
        "targets",
        "issue_ids",
        "test_dates",
        "slopes",
        "deviations",
        "test_prices",
        "ci_widths",
        "ci_lowers",
        "ci_uppers",
        "margins",
        "price_windows",
        "valid_features",
        "valid_targets",
    ]

    def _apply_row_filter(kept_indices: np.ndarray) -> None:
        for field in row_fields:
            if hasattr(dataset, field):
                value = getattr(dataset, field)
                setattr(dataset, field, value[kept_indices])

    _apply_row_filter(kept_indices)

    print(
        "Synthetic class-balanced stats - \n"
        f"share={share:.4f}, n_per_class={requested_n_per_class}, \n"
        f"target_synth_per_class={target_synth}, target_real_per_class={target_real}, \n"
        f"total_after={len(dataset)}, target_total_samples={target_total_samples}, class_stats={class_stats} \n"
    )

    return dataset

def main(trainer_cfg, synthetic_share_of_total, neural_SDE_model_type):

    # Suppress Lightning warning about num_workers=0
    warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck.*")
    multiprocessing.set_start_method("spawn", force=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    wdir = amgm_config.work_dir("neural_SDE")

    dset_cfg = trainer_cfg["dset_cfg"]
    time_bar = dset_cfg["time_bar"]
    w_trn = dset_cfg["window_size_train"]
    w_tst = dset_cfg["window_size_test"]
    trn_dtype = dset_cfg["training_data_type"]
    logger = pl_loggers.TensorBoardLogger(
        name=Path(__file__).stem,
        save_dir=wdir / "logs",
        version=f"{time_bar}_{w_trn}_{w_tst}_{trn_dtype}_{synthetic_share_of_total}_{neural_SDE_model_type}",
    )
    
    print(f"Working directory: {wdir}")
    print(f"Relative log path: {Path(logger.log_dir).relative_to(wdir)}")
    print(f"Full log path: {logger.log_dir}")

    from time import time

    t0 = time()

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        mode="max",                  # We want to maximize accuracy!
        auto_insert_metric_name=False,  # Avoid "val/acc" key be part of the filename, explicitly add the "val_acc" to the filename instead
        filename="best-{epoch:02d}-val_acc={val/acc:.4f}",  # Explicitly named metric val_acc
        save_top_k=1,                # Only save the best
        dirpath=Path(logger.log_dir) / "checkpoints" # Your directory for checkpoints
        )
    
    run_cfg = trainer_cfg["run_cfg"]
    mdl = LinearTrendRunner(**trainer_cfg)
    dmod = LinearTrendDataModule(
        run_cfg,
        mdl.dset_cfg,
        cache_path=wdir / "datasets",
        rebuild_cache=True,
        build_train_dataset=False,
    )
    if synthetic_share_of_total is not None and dmod.train_cfg.get("balancing") == "smallest":
        # Avoid pre-balancing in the dataset; apply class-aware synthetic balancing below in one pass.
        dmod.train_cfg["balancing"] = None

    dmod.train_dataset = _build_augmented_train_dataset(dmod, neural_SDE_model_type)
    print(f"[DEBUG] Train dataset size before augmentation: {len(dmod.train_dataset)} samples.")
    if synthetic_share_of_total is not None:
        dmod.train_dataset = _balance_smallest_with_synthetic_share(
            dmod.train_dataset,
            synthetic_share_of_total=float(synthetic_share_of_total),
        )
    print(f"[DEBUG] Train dataset size after augmentation: {len(dmod.train_dataset)} samples.")
    dmod.train_dataset_file = None
    trainer = Trainer(max_epochs=run_cfg["max_epochs"], check_val_every_n_epoch=1, 
                        logger=False, accelerator="cpu", callbacks=[checkpoint_callback])
    trainer.fit(mdl, datamodule=dmod)
    train_acc_metric = trainer.callback_metrics.get("train/acc")
    train_acc = float(train_acc_metric) * 100 if train_acc_metric is not None else float("nan")
    
    print("Best model score:", checkpoint_callback.best_model_score)
    print(f"Training completed in {time() - t0:.2f} seconds.")

    dmod.rebuild_cache = False  # Disable cache rebuild for validation
    acc = trainer.validate(mdl, datamodule=dmod, verbose=False)

    CM_numpy = mdl.val_confusion_matrix.numpy()
    row_sums = CM_numpy.sum(axis=1, keepdims=True)  # Sum of each row, shape (3, 1)
    # Divide each element by its row sum
    CM_rate = CM_numpy / row_sums * 100
    accuracy = acc[0]["val/acc"] * 100
    
    print(f"Validation Accuracy: {accuracy:.1f}")
    print(f"Confusion matrix:\n{CM_rate}")

    results = trainer.predict(mdl, dataloaders=dmod.val_dataloader())
    merged_results = common.merge_results(results, common.cat_torch, common.cat_numpy)
    y_hat_val, *_ = common.unpack_predict_results(merged_results)
    
    print(f"Validation predictions shape: {y_hat_val.shape}")

    return y_hat_val, CM_rate, float(train_acc), float(accuracy)

def run_single_simulation(param):
    MC_id, trainer_cfg, synthetic_share_of_total, neural_SDE_model_type = param
    trainer_cfg_copy = deepcopy(trainer_cfg)

    second_first_digit = int(f"{datetime.now().second:02d}"[0]) + 1
    seed_everything(int(MC_id) * second_first_digit, workers=True)

    y_hat_val, CF_numpy, train_accuracy, accuracy = main(trainer_cfg_copy, synthetic_share_of_total, neural_SDE_model_type)

    return y_hat_val, CF_numpy, train_accuracy, accuracy

if __name__ == "__main__":
    # This script saves 10 model checkpoints with best val_acc in Path(logger.log_dir) / "checkpoints"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    neural_SDE_model_type = "MoE"   # Options: "MLP" or "MoE" or "PatchTST"
    synthetic_share_of_total = 1.0
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer_cfg",
        help="Dotted module path (relative to this package) exposing get_trainer_cfg().",
    )
    args = parser.parse_args()

    task = Path(__file__).stem.removeprefix("train_")
    cfg_path = f"trainer_cfg.{task}.{args.trainer_cfg}"
    cfg_module = importlib.import_module(cfg_path, package=__package__ or "experiments.linear_trend")
    trainer_cfg = cfg_module.get_trainer_cfg()

    run_cfg = trainer_cfg["run_cfg"]
    dset_cfg = trainer_cfg["dset_cfg"]

    model = dset_cfg["time_bar"] + "_" + str(dset_cfg["window_size_train"]) + "_" + str(dset_cfg["window_size_test"])
    
    num_cpus = multiprocessing.cpu_count()
    print(f"Number of CPUs: {num_cpus}")

    MC_id_list = list(range(1, 11))
    param_combinations = [
        (MC_id, trainer_cfg, synthetic_share_of_total, neural_SDE_model_type)
        for MC_id in MC_id_list
    ]

    with multiprocessing.Pool(processes=10) as pool:
        results = pool.map(run_single_simulation, param_combinations)  # Ordered results

        y_hat_val_list, CM_numpy_list, train_accuracy_list, accuracy_list = zip(*results)

        CM_array = np.array(CM_numpy_list)

        np.set_printoptions(precision=1, suppress=True)
        print(f"CM list: {CM_numpy_list}")
        print(f"Train Accuracy List: {train_accuracy_list}")
        print(f"Accuracy List: {accuracy_list}")
        
        mean_train_accuracy = np.mean(train_accuracy_list)
        std_train_accuracy = np.std(train_accuracy_list)
        mean_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)
        mean_CM = np.mean(CM_array, axis=0)
        std_CM = np.std(CM_array, axis=0)
        
        print(f"Average Train Acc: {mean_train_accuracy:.2f} ± {std_train_accuracy:.2f}%")
        print(f"Average Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"Mean Confusion Matrix:\n{mean_CM}")
        print(f"Std Confusion Matrix:\n{std_CM}")
