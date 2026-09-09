import argparse
import importlib
import multiprocessing
import warnings
from pathlib import Path
from typing import NamedTuple
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import logging
from torch.utils.data import DataLoader, Dataset, random_split
import re

from amgm import config as amgm_config
from amgm.data.neural_SDE import NeuralSDEDataset, SyntheticNeuralSDEDataset
from amgm.models.neural_SDE.runner import NeuralSDERunner
import amgm.utils.common as common
import amgm.utils.myplot as myplot

# For parallel run is necessary otherwise, torch will overload each vCPU 
# torch.set_num_threads(1)          # limit PyTorch to 1 thread for intra-op parallelism
# torch.set_num_interop_threads(1)  # limit inter-op parallelism to 1 thread
torch.set_num_threads(16)
torch.set_num_interop_threads(2)

def _resolve_seed(seed_value):
    if seed_value in (None, "random"):
        return torch.seed() % (2**31 - 1)  
    return int(seed_value)

def _build_dataloaders(dset_cfg, batch_size, seed):

    if dset_cfg.get("training_data_type") == "synthetic_sde":
        dataset = SyntheticNeuralSDEDataset(**dset_cfg, rng_seed=seed)
        is_synthetic_dataset = True
    else:
        dataset = NeuralSDEDataset(**dset_cfg, rng_seed=seed)
        is_synthetic_dataset = False
        
    train_ratio = float(dset_cfg.get("train_split", 0.8))
    train_size = max(1, int(train_ratio * len(dataset)))
    val_size = len(dataset) - train_size

    split_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=split_gen)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, is_synthetic_dataset, seed


def _format_validation_metrics(val_metrics):
    if not val_metrics:
        return "Validation metrics: n/a"

    metrics = val_metrics[0] if isinstance(val_metrics, list) else val_metrics

    pi_mean_values = []
    pi_var_values = []
    other_metrics = []

    def _metric_sort_key(item):
        key, _ = item
        match = re.search(r"_e(\d+)$", key)
        if match:
            return int(match.group(1))
        return key

    def _pretty_number(value):
        value = float(value)
        if abs(value) < 5e-4:
            return "0"
        return f"{value:.3f}"

    for key, value in metrics.items():
        if key.startswith("val/pi_mean_e"):
            pi_mean_values.append((key, value))
        elif key.startswith("val/pi_var_e"):
            pi_var_values.append((key, value))
        else:
            other_metrics.append((key, value))

    pi_mean_values.sort(key=_metric_sort_key)
    pi_var_values.sort(key=_metric_sort_key)
    other_metrics.sort(key=lambda item: item[0])

    lines = ["Validation metrics:"]

    for key, value in other_metrics:
        pretty_key = key.removeprefix("val/")
        if value is None:
            pretty_value = "n/a"
        elif isinstance(value, (float, int)):
            pretty_value = _pretty_number(value)
        else:
            pretty_value = str(value)
        lines.append(f"  {pretty_key}: {pretty_value}")

    if pi_mean_values:
        pi_mean = ", ".join(_pretty_number(value) for _, value in pi_mean_values)
        lines.append(f"  pi = [{pi_mean}]")

    if pi_var_values:
        pi_var = ", ".join(_pretty_number(value) for _, value in pi_var_values)
        lines.append(f"  pi_var = [{pi_var}]")

    return "\n".join(lines)

def main(trainer_cfg, save_rollout_plots, model_type):

    # Suppress Lightning warning about num_workers=0
    warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck.*")
    multiprocessing.set_start_method("spawn", force=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    wdir = amgm_config.work_dir("neural_SDE")

    dset_cfg = trainer_cfg["dset_cfg"]
    run_cfg = trainer_cfg["run_cfg"]
    batch_size = run_cfg["batch_size"]
    seed = _resolve_seed(run_cfg.get("rng_seed"))
    seed_everything(seed, workers=True)
        
    version = run_cfg.get("run_name", "US_Stocks")
    logger = pl_loggers.TensorBoardLogger(
        name=Path(__file__).stem,
        save_dir=wdir / "logs",
        version=version,
    )
    
    checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            auto_insert_metric_name=False,  # Avoid "val/loss" key be part of the filename, explicitly add the "val_loss" to the filename instead
            filename="best-{epoch:02d}-val_loss={val/loss:.4f}_NEW_" + model_type,  # Explicitly named metric val_loss
            save_top_k=1,                # Only save the best
            dirpath=Path(logger.log_dir) / "checkpoints" # Your directory for checkpoints
            )
    
    print(f"Working directory: {wdir}")
    print(f"Relative log path: {Path(logger.log_dir).relative_to(wdir)}")
    print(f"Full log path: {logger.log_dir}")

    from time import time

    t0 = time()

    mdl = NeuralSDERunner(**trainer_cfg)
    train_loader, val_loader, is_synthetic_dataset, seed = _build_dataloaders(dset_cfg, batch_size, seed)
    print(f"Using rng_seed={seed}")

    trainer = Trainer(
        max_epochs=run_cfg["max_epochs"],
        check_val_every_n_epoch=1,
        logger=False,
        accelerator="cpu",
        callbacks=[checkpoint_callback]
    )
    trainer.fit(mdl, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Training completed in {time() - t0:.2f} seconds.")

    val_metrics = trainer.validate(mdl, dataloaders=val_loader, verbose=False)
    predictions = trainer.predict(mdl, dataloaders=val_loader)

    print(_format_validation_metrics(val_metrics))
    if predictions and isinstance(predictions[0], dict):
        first_batch = predictions[0]
        print(f"Prediction batch keys: {list(first_batch.keys())}")
        print(f"Predicted next-price batch shape: {first_batch['x_tp1_pred'].shape}")
        
        if save_rollout_plots:
            # Randomly plot plot_fib_levels for 10 random samples from the validation set
            random_indices = torch.randperm(first_batch["x_window"].shape[0])[:10]
            for idx in random_indices:
                myplot.price_window_and_fib_levels(
                    x_window=first_batch["x_window"][idx].cpu().numpy(),
                    fib_levels=first_batch["fib_levels"][idx].cpu().numpy(),
                    features=first_batch["features"][idx].cpu().numpy(),
                    wdir=wdir,
                    x_t=first_batch["x_t"][idx].item(),
                    x_tp1=first_batch["x_tp1"][idx].item(),
                    x_tp1_pred=first_batch["x_tp1_pred"][idx].item(),
                    file_name=f"price_window_and_fib_levels_{idx.item()}.png"
                )

    residual_metrics = common.evaluate_residual_calibration(predictions, dset_cfg)
    print("Residual calibration metrics:")
    common.print_dict(residual_metrics)

    if is_synthetic_dataset:
        recovery = common.evaluate_synthetic_recovery(predictions, dset_cfg, run_cfg)
        if recovery is not None:
            print("Synthetic recovery metrics:")
            common.print_dict(recovery)

    return val_metrics, predictions

if __name__ == "__main__":
    # This script saves 10 model checkpoints with best val_acc in Path(logger.log_dir) / "checkpoints"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    save_rollout_plots = False
    model_type = "MoE"   # Options: "MLP" or "MoE" or "PatchTST"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer_cfg",
        help="Dotted module path (relative to this package) exposing get_trainer_cfg().",
    )
    args = parser.parse_args()

    if args.trainer_cfg is None:
        raise ValueError("Please provide a trainer configuration module using --trainer_cfg.")
    
    cfg_path = f"trainer_cfg.neural_SDE.{args.trainer_cfg}"
    cfg_module = importlib.import_module(cfg_path, package=__package__ or "experiments.neural_SDE")
    trainer_cfg = cfg_module.get_trainer_cfg()
   
    main(trainer_cfg, save_rollout_plots, model_type)
    
