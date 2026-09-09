import torch
from torch import nn
from torchmetrics import MeanAbsoluteError

from amgm.models.mlp import NeuralSDEMLP

"""Synthetic Neural SDE for Ornstein-Uhlenbeck process recovery test.

Simulated process (Euler-Maruyama, implemented in train.py):
    X_{t+1} = X_t + mu(X_t) * dt + sigma(X_t) * sqrt(dt) * eps_t,
    eps_t ~ N(0, 1)

with
    mu(x) = kappa * (theta - x)
    sigma(x) = sigma_base + sigma_scale * sigmoid(x)
"""


def get_trainer_cfg():
    output_size = 1

    run_cfg = dict(
        run_idx=1,
        run_name="synthetic_ou_recovery", 
        rng_seed=None,  # set to an int for reproducible runs, or None for a fresh random seed each run
        batch_size=128,
        num_workers=0,
        max_epochs=20,
        fail_on_bad_recovery=True,
    )

    dset_cfg = dict(
        training_data_type="synthetic_sde",
        lookback_window=20,
        # dt in: X_{t+1} = X_t + mu(X_t) * dt + sigma(X_t) * sqrt(dt) * eps_t
        dt=1.0 / 252.0,
        num_paths=384,
        steps_per_path=48,
        train_split=0.8,
        x0_mean=0.1,
        x0_std=0.25,
        # mu(x) = kappa * (theta - x)
        kappa=3.0,
        theta=0.0,
        # sigma(x) = sigma_base + sigma_scale * sigmoid(x)
        sigma_base=0.04,
        sigma_scale=0.14,
        # Minimum acceptable recovery correlations on validation samples.
        min_drift_corr=0.85,
        min_diffusion_corr=0.75,
    )

    model_cfg = dict(
        _target_=NeuralSDEMLP,
        lookback_window=dset_cfg["lookback_window"],
        num_features=7,
        hidden_sizes=[64, 32],
        output_size=output_size,
    )

    trainer_cfg = dict(
        run_cfg=run_cfg,
        dset_cfg=dset_cfg,
        model_cfg=model_cfg,
        loss_cfg=dict(_target_=nn.MSELoss),
        acc_cfg=dict(_target_=MeanAbsoluteError),
        optim_cfg=dict(_target_=torch.optim.Adam, lr=5e-4, weight_decay=1e-6),
        sched_cfg=None,
    )

    return trainer_cfg
