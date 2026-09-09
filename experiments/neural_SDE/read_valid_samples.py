import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import amgm.utils.myplot as myplot
from amgm import config as amgm_config


def _require_npz_keys(data, required_keys, mode):
    missing = sorted(set(required_keys) - set(data.files))
    if missing:
        raise ValueError(
            f"{mode} NPZ file is missing required keys: {missing}. "
            f"Available keys: {sorted(data.files)}"
        )


def _load_standard_npz(npz_path: Path) -> dict[str, Any]:
    required_keys = {
        "count",
        "issue_id",
        "test_date",
        "hist",
        "price_min",
        "price_range",
        "fib_levels",
        "original_price",
        "original_price_len",
        "synthetic_path",
    }
    with np.load(npz_path, allow_pickle=False) as data:
        _require_npz_keys(data, required_keys, mode="standard")
        count = int(data["count"][0])
        issue_ids = data["issue_id"]
        test_dates = data["test_date"]
        hist = data["hist"]
        fib_levels = data["fib_levels"]
        synthetic_path = data["synthetic_path"]
        original_price_padded = data["original_price"]
        original_price_len = data["original_price_len"]
        price_min = data["price_min"]
        price_range = data["price_range"]

    samples: list[dict[str, Any]] = []
    for i in range(count):
        n_orig = int(original_price_len[i])
        sample_dict = {
            "issue_id": str(issue_ids[i]),
            "test_date": str(test_dates[i]),
            "hist": hist[i],
            "price_min": float(price_min[i]),
            "price_range": float(price_range[i]),
            "fib_levels": fib_levels[i],
            "original_price": original_price_padded[i, :n_orig],
            "synthetic_path": synthetic_path[i],
        }
        samples.append(sample_dict)

    return {
        "metadata": {
            "source": str(npz_path),
            "mode": "standard",
            "count": count,
            "keys_per_sample": [
                "issue_id",
                "test_date",
                "hist",
                "price_min",
                "price_range",
                "fib_levels",
                "original_price",
                "synthetic_path",
            ],
        },
        "samples": samples,
    }


def _load_mc_npz(npz_path: Path) -> dict[str, Any]:
    common_required_keys = {
        "issue_id",
        "test_date",
        "master_seed",
        "is_valid",
        "hist",
        "price_min",
        "price_range",
        "fib_levels",
        "original_price",
        "synthetic_paths",
    }
    with np.load(npz_path, allow_pickle=False) as data:
        _require_npz_keys(data, common_required_keys, mode="MC")

        synthetic_paths = np.asarray(data["synthetic_paths"])
        if synthetic_paths.ndim != 2:
            raise ValueError(
                "MC synthetic_paths must have shape "
                "(total_paths, horizon + 1), got "
                f"{synthetic_paths.shape}."
            )
        total_paths = synthetic_paths.shape[0]

        # Current multi-initial-condition MC schema.
        if "base_sample_id" in data.files:
            _require_npz_keys(
                data,
                {"base_sample_id", "mc_iteration", "original_price_len"},
                mode="MC",
            )
            base_sample_id = np.asarray(data["base_sample_id"], dtype=np.int32)
            mc_iteration = np.asarray(data["mc_iteration"], dtype=np.int32)
            hist = np.asarray(data["hist"])
            fib_levels = np.asarray(data["fib_levels"])
            original_price = np.asarray(data["original_price"])
            original_price_len = np.asarray(
                data["original_price_len"], dtype=np.int32
            )
            issue_ids = np.asarray(data["issue_id"]).reshape(-1)
            test_dates = np.asarray(data["test_date"]).reshape(-1)
            price_min = np.asarray(data["price_min"]).reshape(-1)
            price_range = np.asarray(data["price_range"]).reshape(-1)
            schema = "multi_initial_condition"

        # Earlier single-initial-condition MC schema.
        elif "path_id" in data.files:
            _require_npz_keys(data, {"path_id"}, mode="MC (legacy)")
            base_sample_id = np.zeros(total_paths, dtype=np.int32)
            mc_iteration = np.asarray(data["path_id"], dtype=np.int32)
            hist = np.asarray(data["hist"])
            if hist.ndim == 1:
                hist = hist[None, :]
            fib_levels = np.asarray(data["fib_levels"])
            if fib_levels.ndim == 1:
                fib_levels = fib_levels[None, :]
            original_price = np.asarray(data["original_price"])
            if original_price.ndim == 1:
                original_price = original_price[None, :]
            original_price_len = np.asarray(
                [np.isfinite(original_price[0]).sum()], dtype=np.int32
            )
            issue_ids = np.asarray(data["issue_id"]).reshape(-1)
            test_dates = np.asarray(data["test_date"]).reshape(-1)
            price_min = np.asarray(data["price_min"]).reshape(-1)
            price_range = np.asarray(data["price_range"]).reshape(-1)
            schema = "single_initial_condition_legacy"
        else:
            raise ValueError(
                "MC NPZ must contain either the current 'base_sample_id' "
                "mapping or the legacy 'path_id' mapping."
            )

        is_valid = np.asarray(data["is_valid"], dtype=bool).reshape(-1)
        master_seed = int(np.asarray(data["master_seed"]).reshape(-1)[0])

    path_arrays = {
        "base_sample_id": base_sample_id,
        "mc_iteration": mc_iteration,
        "is_valid": is_valid,
    }
    for name, values in path_arrays.items():
        if len(values) != total_paths:
            raise ValueError(
                f"MC {name} has length {len(values)}, expected {total_paths}."
            )

    base_sample_count = len(hist)
    if base_sample_count == 0:
        raise ValueError("MC NPZ contains no base samples.")
    if np.any(base_sample_id < 0) or np.any(base_sample_id >= base_sample_count):
        raise ValueError(
            "MC base_sample_id contains indices outside the available "
            f"base-sample range [0, {base_sample_count - 1}]."
        )

    base_arrays = {
        "issue_id": issue_ids,
        "test_date": test_dates,
        "price_min": price_min,
        "price_range": price_range,
        "fib_levels": fib_levels,
        "original_price": original_price,
        "original_price_len": original_price_len,
    }
    for name, values in base_arrays.items():
        if len(values) != base_sample_count:
            raise ValueError(
                f"MC {name} has {len(values)} base samples, "
                f"expected {base_sample_count}."
            )

    samples: list[dict[str, Any]] = []
    for path_idx in range(total_paths):
        base_idx = int(base_sample_id[path_idx])
        n_orig = int(original_price_len[base_idx])
        samples.append(
            {
                "issue_id": str(issue_ids[base_idx]),
                "test_date": str(test_dates[base_idx]),
                "hist": hist[base_idx],
                "price_min": float(price_min[base_idx]),
                "price_range": float(price_range[base_idx]),
                "fib_levels": fib_levels[base_idx],
                "original_price": original_price[base_idx, :n_orig],
                "synthetic_path": synthetic_paths[path_idx],
                "path_id": path_idx,
                "base_sample_id": base_idx,
                "mc_iteration": int(mc_iteration[path_idx]),
                "is_valid": bool(is_valid[path_idx]),
                "master_seed": master_seed,
            }
        )

    unique_issue_ids = np.unique(issue_ids.astype(str))
    iteration_counts = np.bincount(
        base_sample_id, minlength=base_sample_count
    )
    mc_paths_per_sample = (
        int(iteration_counts[0])
        if np.all(iteration_counts == iteration_counts[0])
        else None
    )
    keys_per_sample = list(samples[0].keys()) if samples else []

    return {
        "metadata": {
            "source": str(npz_path),
            "mode": "MC",
            "schema": schema,
            "count": total_paths,
            "base_sample_count": base_sample_count,
            "mc_paths_per_sample": mc_paths_per_sample,
            "issue_id_count": int(len(unique_issue_ids)),
            "master_seed": master_seed,
            "valid_count": int(is_valid.sum()),
            "keys_per_sample": keys_per_sample,
        },
        "samples": samples,
    }


def load_valid_samples_npz(
    npz_path: str | Path,
    mode: str | None = None,
) -> dict[str, Any]:
    """Load saved valid-sample artifacts from NPZ into a Python dictionary.

    Args:
        npz_path: Path to a standard or MC artifact file.
        mode: ``None`` for the standard schema or ``"MC"`` for the Monte
            Carlo schema. The MC value is case-insensitive.

    Returns a dict with two keys:
    - metadata: summary info
    - samples: list[dict] where each dict is one sample
    """
    npz_path = Path(npz_path).expanduser().resolve()
    if mode is None:
        return _load_standard_npz(npz_path)
    if isinstance(mode, str) and mode.casefold() == "mc":
        return _load_mc_npz(npz_path)
    raise ValueError(f"Unsupported mode={mode!r}. Use None or 'MC'.")


def main() -> None:
    
    # Read npz file from Data/Synthetic/synthetic_rollout_MC_valid_samples.npz
    folder_name = "generated_samples_10_steps_100_MoE-Entropy_NEW"
    wdir = amgm_config.work_dir("neural_SDE")
    common_path = wdir / "logs" / "train_neural_SDE" / "US_Stocks"
    output_plot = wdir / folder_name / "synthetic_rollout_reconstructed.png"
    npz_path = wdir / folder_name / "synthetic_rollout_valid_samples.npz"
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    out = load_valid_samples_npz(npz_path)
    print(f"Loaded {out['metadata']['count']} samples from {out['metadata']['source']}")

    if out["samples"]:
        s0 = out["samples"][0]
        sample_obj = SimpleNamespace(
            price_window=torch.from_numpy(np.asarray(s0["hist"], dtype=np.float32)),
            sample_min=torch.tensor(s0["price_min"], dtype=torch.float32),
            sample_range=torch.tensor(s0["price_range"], dtype=torch.float32),
            fib_levels=torch.from_numpy(np.asarray(s0["fib_levels"], dtype=np.float32)),
            test_dates=s0["test_date"],
            issue_ids=s0["issue_id"],
        )

        myplot.synthetic_NSDE_path(
            sample_obj,
            s0["synthetic_path"],
            s0["original_price"],
            wdir=wdir,
            output_plot=output_plot,
        )
        print(f"Saved first sample plot to: {output_plot.parent}")


if __name__ == "__main__":
    main()
