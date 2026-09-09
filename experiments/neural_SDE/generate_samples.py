import argparse
import importlib
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from lightning import seed_everything
import logging
import matplotlib.pyplot as plt
import time

from amgm.data.neural_SDE import NeuralSDEDataset
from amgm.models.neural_SDE.runner import NeuralSDERunner
import amgm.utils.common as common
from amgm import config as amgm_config
import amgm.utils.myplot as myplot
from amgm.data.loading import load_sebx_am_data

# For parallel run is necessary otherwise, torch will overload each vCPU 
# torch.set_num_threads(1)          # limit PyTorch to 1 thread for intra-op parallelism
# torch.set_num_interop_threads(1)  # limit inter-op parallelism to 1 thread
torch.set_num_threads(16)
torch.set_num_interop_threads(2)

def _resolve_seed(seed_value):
    if seed_value in (None, "random"):
        return torch.seed() % (2**31 - 1)  
    return int(seed_value)


def _load_model_checkpoint(checkpoint_path):
    """Load a runner, including legacy MoE checkpoints using `backbone`."""
    checkpoint_path = str(Path(checkpoint_path).expanduser())
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise RuntimeError("Cannot load checkpoint without hyper_parameters.")

    model = NeuralSDERunner(**hparams, compile_model=False)
    state_dict = checkpoint.get("state_dict", {})
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint state_dict is missing or invalid.")

    migrated_state = {}
    for key, value in state_dict.items():
        migrated_key = key
        if migrated_key.startswith("model._orig_mod."):
            migrated_key = migrated_key.replace("model._orig_mod.", "model.", 1)
        if migrated_key.startswith("model.backbone."):
            migrated_key = migrated_key.replace("model.backbone.", "model.encoder.", 1)
        migrated_state[migrated_key] = value

    model.load_state_dict(migrated_state, strict=True)
    if any(key.startswith("model._orig_mod.") for key in state_dict):
        logging.warning(
            "Loaded compiled checkpoint after stripping model._orig_mod.* prefixes."
        )
    if any(key.startswith("model.backbone.") for key in state_dict):
        logging.warning(
            "Loaded legacy MoE checkpoint after migrating model.backbone.* keys to model.encoder.*."
        )
    return model


def _build_dataset(dset_cfg, seed, batch_size): 
    dataset = NeuralSDEDataset(**dset_cfg, rng_seed=seed)
    L = len(dataset)    
    if batch_size > L:
        raise IndexError(f"Requested batch of size {batch_size} exceeds dataset size {L}.")
    
    batch = {
        "price_window": torch.stack([dataset[i].price_window for i in range(L)], dim=0),
        "sample_min": torch.stack([dataset[i].sample_min for i in range(L)], dim=0),
        "sample_range": torch.stack([dataset[i].sample_range for i in range(L)], dim=0),
    }

    return dataset, batch, seed

def _compute_features(x_window):
    x_window = x_window.detach().cpu().numpy()
    fib_levels, delta = common.calculate_fibLevels(x_window)
    x_t = x_window[:, -1:]
    features = (x_t - fib_levels) / np.maximum(delta, 1e-8)
    return torch.from_numpy(features), fib_levels

def _compute_min_range(x_window):
    x_min = x_window.amin(dim=1, keepdim=True)
    x_max = x_window.amax(dim=1, keepdim=True)
    x_range = torch.clamp(x_max - x_min, min=1e-8)
    return x_min, x_range

def _rollout_sde(model, batch, n_steps, dt, seed):
    gen = torch.Generator().manual_seed(int(seed))

    x_window_norm0 = batch["price_window"].detach().clone().float()
    x_min0 = batch["sample_min"].detach().clone().float().view(-1, 1)
    x_range0 = batch["sample_range"].detach().clone().float().view(-1, 1)
    x_window_original = x_window_norm0 * x_range0 + x_min0

    # Keep one rollout per sample: shape [batch_size, n_steps + 1]
    synthetic_path = [x_window_original[:, -1].detach().cpu().numpy()]  # For continuous plot visualization
    pi_path = []
    mu_path = []
    sigma_path = []
    with torch.no_grad():
        for step_idx in range(n_steps):
            # Re-normalize each rolling window before inference, matching training-time preprocessing.
            x_min, x_range = _compute_min_range(x_window_original)
            x_window = (x_window_original - x_min) / x_range

            f_t, _ = _compute_features(x_window)
            mu, sigma, pi = model(x_window, f_t)

            noise = torch.randn(mu.shape, generator=gen, dtype=mu.dtype, device=mu.device)
            x_next_norm = x_window[:, -1:] + mu * dt + sigma * (dt ** 0.5) * noise
            x_next_original = x_next_norm * x_range + x_min

            synthetic_path.append(x_next_original[:, 0].detach().cpu().numpy())
            pi_path.append(pi.detach().cpu().numpy())
            mu_path.append((mu * x_range).detach().cpu().numpy())
            sigma_path.append((sigma * x_range).detach().cpu().numpy())
            x_window_original = torch.cat([x_window_original[:, 1:], x_next_original], dim=1)

    return (
            np.stack(synthetic_path, axis=1).astype(np.float32),
            np.stack(pi_path, axis=1).astype(np.float32),
            np.stack(mu_path, axis=1).astype(np.float32),
            np.stack(sigma_path, axis=1).astype(np.float32),
        )
    
def fetch_original_price(sample, security_data, n_steps):
    """ Fetch the original price data for a given sample from the security_data DataFrame for the next n_steps.
    """
    test_date = pd.Timestamp(str(sample.test_dates))
    issue_id = sample.issue_ids
    issue_rows = security_data.loc[
        security_data["IssueId"] == issue_id,
        ["Date", "ClAdjLoc"],
    ].sort_values("Date").reset_index(drop=True)
    test_matches = issue_rows.index[issue_rows["Date"] == test_date]
    if len(test_matches) == 0:
        raise ValueError(f"test_date={test_date} not found for IssueId={issue_id}")

    test_idx = int(test_matches[0])
    end_idx = min(len(issue_rows), test_idx + 1 + n_steps)
    print(
        f"Filtering security_data for IssueId={issue_id}, test_date={test_date}, "
        f"row_window=[{test_idx}:{end_idx}), test_idx={test_idx}"
    )
    original_price = issue_rows.iloc[test_idx:end_idx]["ClAdjLoc"]
    return original_price


def _save_valid_sample_artifacts(records, output_plot):
    """Persist compact per-sample artifacts for keep_sample=True entries."""
    if not records:
        return None

    output_plot = Path(output_plot)
    artifacts_file = output_plot.with_name(f"{output_plot.stem}_valid_samples.npz")

    payload = {}
    n = len(records)
    payload["count"] = np.array([n], dtype=np.int32)

    payload["issue_id"] = np.asarray([str(r["issue_id"]) for r in records], dtype=str)
    payload["test_date"] = np.asarray([str(r["test_date"]) for r in records], dtype=str)

    payload["hist"] = np.stack([np.asarray(r["hist"], dtype=np.float32) for r in records], axis=0)
    payload["fib_levels"] = np.stack([np.asarray(r["fib_levels"], dtype=np.float32) for r in records], axis=0)
    payload["synthetic_path"] = np.stack([np.asarray(r["synthetic_path"], dtype=np.float32) for r in records], axis=0)

    original_price_arrs = [np.asarray(r["original_price"], dtype=np.float32) for r in records]
    original_price_len = np.asarray([arr.shape[0] for arr in original_price_arrs], dtype=np.int32)
    max_original_price_len = int(original_price_len.max())
    original_price_padded = np.full((n, max_original_price_len), np.nan, dtype=np.float32)
    for idx, arr in enumerate(original_price_arrs):
        original_price_padded[idx, : arr.shape[0]] = arr
    payload["original_price"] = original_price_padded
    payload["original_price_len"] = original_price_len

    payload["price_min"] = np.array([float(r["price_min"]) for r in records], dtype=np.float32)
    payload["price_range"] = np.array([float(r["price_range"]) for r in records], dtype=np.float32)

    np.savez_compressed(artifacts_file, **payload)
    return artifacts_file


def _select_mc_sample(dataset, issue_id, test_date):
    """Return the unique dataset sample matching an issue ID and test date."""
    target_issue_id = str(issue_id)
    target_date = str(pd.Timestamp(test_date).date())
    matches = [
        idx
        for idx in range(len(dataset))
        if str(dataset[idx].issue_ids) == target_issue_id
        and str(pd.Timestamp(str(dataset[idx].test_dates)).date()) == target_date
    ]

    if not matches:
        available_dates = [
            str(dataset[idx].test_dates)
            for idx in range(len(dataset))
            if str(dataset[idx].issue_ids) == target_issue_id
        ]
        date_hint = (
            f" Available range: {min(available_dates)} to {max(available_dates)}."
            if available_dates
            else ""
        )
        raise ValueError(
            f"No sample found for issue_id={target_issue_id}, test_date={target_date}."
            f"{date_hint}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Expected one sample for issue_id={target_issue_id}, "
            f"test_date={target_date}, found {len(matches)}."
        )
    return dataset[matches[0]]


def _build_mc_batch(sample, mc_paths):
    """Repeat one initial condition for vectorized Monte Carlo generation."""
    if mc_paths <= 0:
        raise ValueError(f"mc_paths must be positive, got {mc_paths}.")

    return {
        "price_window": sample.price_window.detach().clone().float().unsqueeze(0).repeat(mc_paths, 1),
        "sample_min": sample.sample_min.detach().clone().float().reshape(1).repeat(mc_paths),
        "sample_range": sample.sample_range.detach().clone().float().reshape(1).repeat(mc_paths),
    }


def _save_mc_artifacts(
    output_plot,
    sample,
    synthetic_paths,
    original_price,
    is_valid,
    master_seed,
):
    """Save shared initial data and all Monte Carlo paths without duplication."""
    output_plot = Path(output_plot)
    artifacts_file = output_plot.with_name(f"{output_plot.stem}_valid_samples.npz")

    np.savez_compressed(
        artifacts_file,
        issue_id=np.asarray([str(sample.issue_ids)], dtype=str),
        test_date=np.asarray([str(sample.test_dates)], dtype=str),
        master_seed=np.asarray([int(master_seed)], dtype=np.int64),
        path_id=np.arange(len(synthetic_paths), dtype=np.int32),
        is_valid=np.asarray(is_valid, dtype=bool),
        hist=sample.price_window.detach().cpu().numpy().astype(np.float32),
        price_min=np.asarray(
            [float(sample.sample_min.detach().cpu().item())], dtype=np.float32
        ),
        price_range=np.asarray(
            [float(sample.sample_range.detach().cpu().item())], dtype=np.float32
        ),
        fib_levels=sample.fib_levels.detach().cpu().numpy().astype(np.float32),
        original_price=np.asarray(original_price, dtype=np.float32),
        synthetic_paths=np.asarray(synthetic_paths, dtype=np.float32),
    )
    return artifacts_file


def _plot_mc_ensemble(sample, synthetic_paths, original_price, output_plot):
    """Plot the MC ensemble, median, and 5--95% interval."""
    output_plot = Path(output_plot)
    hist = sample.price_window.detach().cpu().numpy()
    price_min = float(sample.sample_min.detach().cpu().item())
    price_range = float(sample.sample_range.detach().cpu().item())
    hist = hist * price_range + price_min

    hist_idx = np.arange(len(hist))
    future_idx = np.arange(
        len(hist) - 1,
        len(hist) - 1 + synthetic_paths.shape[1],
    )
    median_path = np.median(synthetic_paths, axis=0)
    lower_path, upper_path = np.quantile(synthetic_paths, [0.05, 0.95], axis=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hist_idx, hist, color="black", linewidth=2, label="seed window")
    for path_idx, path in enumerate(synthetic_paths):
        ax.plot(
            future_idx,
            path,
            color="tab:blue",
            linewidth=0.9,
            alpha=0.18,
            label="MC paths" if path_idx == 0 else None,
        )
    ax.fill_between(
        future_idx,
        lower_path,
        upper_path,
        color="tab:blue",
        alpha=0.16,
        label="MC 5--95% interval",
    )
    ax.plot(
        future_idx,
        median_path,
        color="navy",
        linewidth=2.2,
        label="MC median",
    )

    original_price = np.asarray(original_price, dtype=np.float32)
    original_len = min(len(future_idx), len(original_price))
    if original_len:
        ax.plot(
            future_idx[:original_len],
            original_price[:original_len],
            color="tab:green",
            linewidth=2,
            label="original price",
        )

    ax.scatter(
        [future_idx[0]],
        [synthetic_paths[0, 0]],
        color="tab:red",
        zorder=5,
        label="SDE start",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title(
        f"Neural SDE Monte Carlo ensemble, Issue_id: {sample.issue_ids}, "
        f"date: {sample.test_dates}"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_issue_id_for_filename(issue_id):
    """Preserve the conventional eight-character numeric issue-ID format."""
    issue_id = str(issue_id)
    return issue_id.zfill(8) if issue_id.isdigit() else issue_id


def _plot_mc_single_path(sample, synthetic_path, original_price, output_file):
    """Save one MC path using the same visual structure as standard plots."""
    hist = sample.price_window.detach().cpu().numpy()
    price_min = float(sample.sample_min.detach().cpu().item())
    price_range = float(sample.sample_range.detach().cpu().item())
    fib_levels = sample.fib_levels.detach().cpu().numpy()
    hist = hist * price_range + price_min
    fib_levels = fib_levels * price_range + price_min

    hist_idx = np.arange(len(hist))
    future_idx = np.arange(
        len(hist) - 1,
        len(hist) - 1 + len(synthetic_path),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_idx, hist, label="seed window", color="black", linewidth=2)
    ax.plot(
        future_idx,
        synthetic_path,
        label="synthetic rollout",
        color="tab:blue",
        linewidth=2,
    )
    ax.scatter(
        [future_idx[0]],
        [synthetic_path[0]],
        color="tab:red",
        zorder=5,
        label="SDE start",
    )

    original_price = np.asarray(original_price, dtype=np.float32)
    original_len = min(len(future_idx), len(original_price))
    if original_len:
        ax.plot(
            future_idx[:original_len],
            original_price[:original_len],
            label="original price",
            color="tab:green",
            linewidth=2,
        )

    myplot.plot_fib_levels(fib_levels, ax)
    ax.set_xlim(left=0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title(
        f"Neural SDE MC rollout, Issue_id: "
        f"{_format_issue_id_for_filename(sample.issue_ids)}, "
        f"date: {sample.test_dates}"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main_mc(
    trainer_cfg,
    checkpoint_path,
    issue_id,
    test_date,
    mc_paths,
    n_steps,
    output_plot,
    seed_override=None,
):
    """Generate multiple MC paths from one issue/date initial condition."""
    run_cfg = trainer_cfg["run_cfg"]
    dset_cfg = dict(trainer_cfg["dset_cfg"])
    dset_cfg["issue_ids"] = [str(issue_id)]
    dset_cfg["num_iids"] = 1
    # Do not truncate windows before the requested date can be selected.
    dset_cfg["max_windows_per_issue"] = None

    configured_seed = seed_override if seed_override is not None else run_cfg.get("rng_seed")
    seed = _resolve_seed(configured_seed)
    seed_everything(seed, workers=True)

    dataset = NeuralSDEDataset(**dset_cfg, rng_seed=seed)
    sample = _select_mc_sample(dataset, issue_id=issue_id, test_date=test_date)
    batch = _build_mc_batch(sample, mc_paths=mc_paths)
    print(
        f"MC mode: issue_id={sample.issue_ids}, test_date={sample.test_dates}, "
        f"mc_paths={mc_paths}, n_steps={n_steps}, master_seed={seed}"
    )

    model = _load_model_checkpoint(checkpoint_path)
    model.eval()

    dt = float(dset_cfg.get("dt", 1.0))
    synthetic_paths, pi_paths, mu_paths, sigma_paths = _rollout_sde(
        model,
        batch,
        n_steps=n_steps,
        dt=dt,
        seed=seed,
    )
    is_valid = np.asarray(
        [bool(common.sanity_check_path(path)) for path in synthetic_paths],
        dtype=bool,
    )

    raw_dataset = load_sebx_am_data(amgm_config.am_dataset_dir)
    original_price = fetch_original_price(
        sample,
        raw_dataset["security_data"],
        n_steps,
    ).to_numpy(dtype=np.float32)

    output_plot = Path(output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(
        start=pd.Timestamp(str(sample.test_dates)),
        periods=n_steps + 1,
    )
    frames = []
    for path_idx, path in enumerate(synthetic_paths):
        frames.append(
            pd.DataFrame(
                {
                    "IssueId": [
                        f"{sample.issue_ids}_synthetic_MC_{path_idx:03d}"
                    ]
                    * (n_steps + 1),
                    "SourceIssueId": [str(sample.issue_ids)] * (n_steps + 1),
                    "TestDate": [str(sample.test_dates)] * (n_steps + 1),
                    "PathId": [path_idx] * (n_steps + 1),
                    "MasterSeed": [int(seed)] * (n_steps + 1),
                    "IsValid": [bool(is_valid[path_idx])] * (n_steps + 1),
                    "Date": dates,
                    "ClAdjLoc": path,
                }
            )
        )

    synthetic_output_file = output_plot.with_name(f"{output_plot.stem}_paths.csv")
    pd.concat(frames, ignore_index=True).to_csv(synthetic_output_file, index=False)
    artifacts_file = _save_mc_artifacts(
        output_plot=output_plot,
        sample=sample,
        synthetic_paths=synthetic_paths,
        original_price=original_price,
        is_valid=is_valid,
        master_seed=seed,
    )
    _plot_mc_ensemble(
        sample=sample,
        synthetic_paths=synthetic_paths,
        original_price=original_price,
        output_plot=output_plot,
    )

    print(f"Saved MC paths CSV to: {synthetic_output_file}")
    print(f"Saved MC artifacts NPZ to: {artifacts_file}")
    print(f"Saved MC ensemble plot to: {output_plot}")
    print(f"Sanity-valid MC paths: {int(is_valid.sum())}/{mc_paths}")
    return synthetic_paths


def main_mc_multi(
    trainer_cfg,
    checkpoint_path,
    batch_size,
    mc_paths,
    n_steps,
    output_plot,
    num_iids,
    issue_id=None,
    test_date=None,
    seed_override=None,
):
    """Generate MC ensembles for several issue/date initial conditions."""
    if batch_size <= 0 or mc_paths <= 0:
        raise ValueError("batch_size and mc_paths must be positive.")

    run_cfg = trainer_cfg["run_cfg"]
    dset_cfg = dict(trainer_cfg["dset_cfg"])
    if issue_id is not None:
        dset_cfg["issue_ids"] = [str(issue_id)]
        dset_cfg["num_iids"] = 1
    else:
        dset_cfg["issue_ids"] = []
        dset_cfg["num_iids"] = int(num_iids)
    dset_cfg["max_windows_per_issue"] = None

    configured_seed = seed_override if seed_override is not None else run_cfg.get("rng_seed")
    seed = _resolve_seed(configured_seed)
    seed_everything(seed, workers=True)
    dataset = NeuralSDEDataset(**dset_cfg, rng_seed=seed)

    target_date = (
        str(pd.Timestamp(test_date).date()) if test_date is not None else None
    )
    candidates = [
        idx
        for idx in range(len(dataset))
        if (issue_id is None or str(dataset[idx].issue_ids) == str(issue_id))
        and (
            target_date is None
            or str(pd.Timestamp(str(dataset[idx].test_dates)).date()) == target_date
        )
    ]
    if not candidates:
        raise ValueError(
            "No MC initial conditions matched the requested issue/date filters."
        )

    # Select distinct starting samples, spreading them across issue IDs first.
    generator = torch.Generator().manual_seed(int(seed))
    shuffled = torch.randperm(len(candidates), generator=generator).tolist()
    candidates = [candidates[pos] for pos in shuffled]
    candidates_by_issue = {}
    for idx in candidates:
        candidates_by_issue.setdefault(str(dataset[idx].issue_ids), []).append(idx)

    requested_initial_conditions = int(np.ceil(batch_size / mc_paths))
    selected_indices = []
    issue_order = list(candidates_by_issue)
    while len(selected_indices) < requested_initial_conditions:
        added = False
        for iid in issue_order:
            if candidates_by_issue[iid]:
                selected_indices.append(candidates_by_issue[iid].pop())
                added = True
                if len(selected_indices) >= requested_initial_conditions:
                    break
        if not added:
            break

    if len(selected_indices) < requested_initial_conditions:
        logging.warning(
            "Only %s distinct issue/date samples are available; requested %s. "
            "The MC output will contain %s paths instead of %s.",
            len(selected_indices),
            requested_initial_conditions,
            len(selected_indices) * mc_paths,
            batch_size,
        )

    base_samples = [dataset[idx] for idx in selected_indices]
    path_samples = []
    mc_iteration = []
    base_sample_index = []
    for base_idx, sample in enumerate(base_samples):
        for iteration in range(mc_paths):
            if len(path_samples) >= batch_size:
                break
            path_samples.append(sample)
            mc_iteration.append(iteration)
            base_sample_index.append(base_idx)

    batch = {
        "price_window": torch.stack(
            [sample.price_window for sample in path_samples], dim=0
        ).float(),
        "sample_min": torch.stack(
            [sample.sample_min for sample in path_samples], dim=0
        ).float(),
        "sample_range": torch.stack(
            [sample.sample_range for sample in path_samples], dim=0
        ).float(),
    }
    print(
        f"MC mode: issue_ids={len(set(str(s.issue_ids) for s in base_samples))}, "
        f"initial_conditions={len(base_samples)}, mc_paths_per_condition={mc_paths}, "
        f"total_paths={len(path_samples)}, n_steps={n_steps}, master_seed={seed}"
    )

    model = _load_model_checkpoint(checkpoint_path)
    model.eval()
    synthetic_paths, pi_paths, mu_paths, sigma_paths = _rollout_sde(
        model,
        batch,
        n_steps=n_steps,
        dt=float(dset_cfg.get("dt", 1.0)),
        seed=seed,
    )
    is_valid = np.asarray(
        [bool(common.sanity_check_path(path)) for path in synthetic_paths],
        dtype=bool,
    )

    security_data = load_sebx_am_data(amgm_config.am_dataset_dir)["security_data"]
    original_prices = [
        fetch_original_price(sample, security_data, n_steps).to_numpy(
            dtype=np.float32
        )
        for sample in base_samples
    ]

    output_plot = Path(output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for path_idx, (sample, path) in enumerate(zip(path_samples, synthetic_paths)):
        dates = pd.bdate_range(
            start=pd.Timestamp(str(sample.test_dates)),
            periods=n_steps + 1,
        )
        frames.append(
            pd.DataFrame(
                {
                    "IssueId": [
                        f"{sample.issue_ids}_synthetic_MC_"
                        f"{base_sample_index[path_idx]:04d}_{mc_iteration[path_idx]:03d}"
                    ]
                    * (n_steps + 1),
                    "SourceIssueId": [str(sample.issue_ids)] * (n_steps + 1),
                    "TestDate": [str(sample.test_dates)] * (n_steps + 1),
                    "BaseSampleId": [base_sample_index[path_idx]] * (n_steps + 1),
                    "PathId": [path_idx] * (n_steps + 1),
                    "MCIteration": [mc_iteration[path_idx]] * (n_steps + 1),
                    "MasterSeed": [int(seed)] * (n_steps + 1),
                    "IsValid": [bool(is_valid[path_idx])] * (n_steps + 1),
                    "Date": dates,
                    "ClAdjLoc": path,
                }
            )
        )

    synthetic_output_file = output_plot.with_name(f"{output_plot.stem}_paths.csv")
    pd.concat(frames, ignore_index=True).to_csv(synthetic_output_file, index=False)

    original_lengths = np.asarray(
        [len(values) for values in original_prices], dtype=np.int32
    )
    original_padded = np.full(
        (len(original_prices), int(original_lengths.max())),
        np.nan,
        dtype=np.float32,
    )
    for idx, values in enumerate(original_prices):
        original_padded[idx, : len(values)] = values

    synthetic_artifacts_dir = (
        Path(__file__).resolve().parents[2] / "Data" / "Synthetic"
    )
    synthetic_artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_file = synthetic_artifacts_dir / f"{output_plot.stem}_valid_samples.npz"
    np.savez_compressed(
        artifacts_file,
        issue_id=np.asarray([str(s.issue_ids) for s in base_samples], dtype=str),
        test_date=np.asarray([str(s.test_dates) for s in base_samples], dtype=str),
        master_seed=np.asarray([int(seed)], dtype=np.int64),
        base_sample_id=np.asarray(base_sample_index, dtype=np.int32),
        mc_iteration=np.asarray(mc_iteration, dtype=np.int32),
        is_valid=is_valid,
        hist=np.stack(
            [s.price_window.detach().cpu().numpy() for s in base_samples]
        ).astype(np.float32),
        price_min=np.asarray(
            [float(s.sample_min.item()) for s in base_samples], dtype=np.float32
        ),
        price_range=np.asarray(
            [float(s.sample_range.item()) for s in base_samples], dtype=np.float32
        ),
        fib_levels=np.stack(
            [s.fib_levels.detach().cpu().numpy() for s in base_samples]
        ).astype(np.float32),
        original_price=original_padded,
        original_price_len=original_lengths,
        synthetic_paths=synthetic_paths,
    )

    # Save one ensemble figure per issue/date initial condition. This keeps all
    # MC realizations comparable without creating one image per path.
    base_sample_index_array = np.asarray(base_sample_index)
    ensemble_outputs = []
    for base_idx, sample in enumerate(base_samples):
        formatted_issue_id = _format_issue_id_for_filename(sample.issue_ids)
        formatted_date = str(pd.Timestamp(str(sample.test_dates)).date())
        ensemble_output = output_plot.with_name(
            f"synthetic_rollout_{formatted_issue_id}_{formatted_date}_MC.png"
        )
        _plot_mc_ensemble(
            sample=sample,
            synthetic_paths=synthetic_paths[base_sample_index_array == base_idx],
            original_price=original_prices[base_idx],
            output_plot=ensemble_output,
        )
        ensemble_outputs.append(ensemble_output)

    print(f"Saved MC paths CSV to: {synthetic_output_file}")
    print(f"Saved MC artifacts NPZ to: {artifacts_file}")
    print(
        f"Saved {len(ensemble_outputs)} MC ensemble plots under: "
        f"{output_plot.parent}"
    )
    print(f"Sanity-valid MC paths: {int(is_valid.sum())}/{len(is_valid)}")
    return synthetic_paths


def main(trainer_cfg, checkpoint_path, batch_size, n_steps, output_plot, wdir, model_type):
    run_cfg = trainer_cfg["run_cfg"]
    dset_cfg = trainer_cfg["dset_cfg"]

    seed = _resolve_seed(run_cfg.get("rng_seed"))
    seed_everything(seed, workers=True)

    dataset, batch, seed = _build_dataset(dset_cfg, seed, batch_size)
    print(f"Using rng_seed={seed}, batch_size={batch_size}, n_steps={n_steps}")

    model = _load_model_checkpoint(checkpoint_path)
    model.eval()

    dt = float(dset_cfg.get("dt", 1.0))
    synthetic_paths,  pi_paths, mu_paths, sigma_paths = _rollout_sde(model, batch, n_steps=n_steps, dt=dt, seed=seed)
    
    raw_dataset = load_sebx_am_data(amgm_config.am_dataset_dir)
    security_data = raw_dataset["security_data"]
    
    output_plot = Path(output_plot)
    synthetic_frames = []
    valid_sample_artifacts = []
    shuffled_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    for i, sample_idx in enumerate(shuffled_indices):
        if len(synthetic_frames) >= batch_size:
            break
        sample = dataset[sample_idx]
        synthetic_path = synthetic_paths[sample_idx]
        keep_sample = common.sanity_check_path(synthetic_path)
        if keep_sample:
            original_price = fetch_original_price(sample, security_data, n_steps)

            hist = sample.price_window.detach().cpu().numpy().astype(np.float32)
            price_min = float(sample.sample_min.detach().cpu().item())
            price_range = float(sample.sample_range.detach().cpu().item())
            fib_levels = sample.fib_levels.detach().cpu().numpy().astype(np.float32)
            test_date = sample.test_dates
            issue_id = sample.issue_ids

            valid_sample_artifacts.append(
                {
                    "issue_id": issue_id,
                    "test_date": test_date,
                    "hist": hist,
                    "price_min": price_min,
                    "price_range": price_range,
                    "fib_levels": fib_levels,
                    "original_price": original_price.to_numpy(dtype=np.float32),
                    "synthetic_path": synthetic_path.astype(np.float32),
                }
            )

            synthetic_iid = f"{sample.issue_ids}_synthetic_{i}"
            synthetic_frames.append(
                pd.DataFrame({  "IssueId": [synthetic_iid] * (n_steps + 1),
                                "Date": pd.bdate_range(start=pd.Timestamp(str(sample.test_dates)), periods=n_steps + 1),
                                "ClAdjLoc": synthetic_path,}))
            if save_rollout_plots:
                myplot.synthetic_NSDE_path(sample, synthetic_path, original_price, wdir=wdir, output_plot=output_plot)
        else:
            print(f"Sanity check for sample {i} completed with result: {keep_sample}")

    synthetic_df = pd.concat(synthetic_frames, ignore_index=True)
    synthetic_output_dir = Path(__file__).resolve().parents[2] / "Data" / "Synthetic"
    synthetic_output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_output_file = synthetic_output_dir / f"{output_plot.stem}_paths_{model_type}.csv"
    synthetic_df.to_csv(synthetic_output_file, index=False)
    valid_samples_file = _save_valid_sample_artifacts(valid_sample_artifacts, output_plot)
    
    print(f"Saved synthetic paths .csv to: {synthetic_output_file}")
    if valid_samples_file is not None:
        print(f"Saved valid sample artifacts .npz to: {valid_samples_file}")
    print(f"Saved {len(synthetic_frames)} out of {len(synthetic_paths)} synthetic rollouts under: {output_plot.parent}")

    if len(synthetic_frames) < batch_size:
        logging.warning(
            "Only %s valid synthetic paths found out of requested %s. "
            "Try increasing dataset size or relaxing sanity checks.",
            len(synthetic_frames),
            batch_size,
        )
        
    return synthetic_paths

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate Neural SDE samples. MC mode uses defaults of 1,000 total "
            "paths, 20 paths per initial condition, 100 forecast steps, and a "
            "reduced issue-ID set."
        ),
        epilog=(
            "MC example: python experiments/neural_SDE/generate_samples.py "
            "--trainer_cfg US_Stocks --mode mc"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trainer_cfg",
        default="US_Stocks",
        help="Dotted module path (relative to this package) exposing get_trainer_cfg().",
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "mc"),
        default="standard",
        help="Use 'mc' to generate repeated paths from multiple initial samples.",
    )
    parser.add_argument("--issue-id", "--issue_id", dest="issue_id")
    parser.add_argument("--test-date", "--test_date", dest="test_date")
    parser.add_argument("--mc-paths", "--mc_paths", dest="mc_paths", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument(
        "--num-iids",
        type=int,
        help=(
            "Override the issue-ID count. MC defaults to one fifth of the "
            "trainer configuration, with a minimum of two."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override run_cfg.rng_seed. Recommended for reproducible MC runs.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best-06-val_loss=-2.6836_MoE" + ".ckpt",
    )
    parser.add_argument(
        "--save-rollout-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save individual rollout plots in standard mode.",
    )
    args = parser.parse_args()

    cfg_path = f"trainer_cfg.neural_SDE.{args.trainer_cfg}"
    cfg_module = importlib.import_module(cfg_path, package=__package__ or "experiments.neural_SDE")
    trainer_cfg = cfg_module.get_trainer_cfg()

    if args.seed is not None:
        trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"])
        trainer_cfg["run_cfg"]["rng_seed"] = args.seed
    if args.num_iids is not None:
        trainer_cfg["dset_cfg"] = dict(trainer_cfg["dset_cfg"])
        trainer_cfg["dset_cfg"]["num_iids"] = args.num_iids

    batch_size = args.batch_size
    n_steps = args.n_steps
    checkpoint_name = args.checkpoint_name + (".ckpt" if not args.checkpoint_name.endswith(".ckpt") else "")
    model_type = checkpoint_name.split("_")[-1].split(".")[0]
    save_rollout_plots = args.save_rollout_plots

    wdir = amgm_config.work_dir("neural_SDE")
    common_path = wdir / "logs" / "train_neural_SDE" / "US_Stocks"
    checkpoint_path = common_path / "checkpoints" / checkpoint_name

    if args.mode == "mc":
        filter_suffix = ""
        if args.issue_id is not None:
            filter_suffix += f"_{str(args.issue_id).replace('/', '-')}"
        if args.test_date is not None:
            filter_suffix += f"_{str(pd.Timestamp(args.test_date).date())}"
        output_dir = wdir / (
            f"generated_samples_{batch_size}_paths_{args.mc_paths}_MC_"
            f"{n_steps}_steps_{model_type}{filter_suffix}"
        )
        output_plot = output_dir / "synthetic_rollout_MC.png"
    else:
        output_dir = wdir / (
            f"generated_samples_{batch_size}_steps_{n_steps}_{model_type}"
        )
        output_plot = output_dir / "synthetic_rollout.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    # time counter starts 
    start_time = time.time()
    if args.mode == "mc":
        configured_num_iids = int(
            trainer_cfg["dset_cfg"].get("num_iids", 50)
        )
        mc_num_iids = (
            args.num_iids
            if args.num_iids is not None
            else max(2, configured_num_iids)
        )
        main_mc_multi(
            trainer_cfg,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            mc_paths=args.mc_paths,
            num_iids=mc_num_iids,
            issue_id=args.issue_id,
            test_date=args.test_date,
            n_steps=n_steps,
            output_plot=output_plot,
            seed_override=args.seed,
        )
    else:
        main(
            trainer_cfg,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            n_steps=n_steps,
            output_plot=output_plot,
            wdir=wdir,
            model_type=model_type,
        )
    # time counter ends
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
