import numpy as np
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError

from amgm import config as amgm_config
from amgm.models.mlp import NeuralSDEMLP, NeuralSDEMoE, PatchNeuralSDETransformer

def get_trainer_cfg():
    output_size = 1

    run_cfg = dict(
        run_idx=1,
        rng_seed=None,  # set an int for reproducible issue-id sampling; None gives a fresh subset each run
        batch_size=256,
        num_workers=0,
        max_epochs=20,
    )

    dset_cfg = dict(
        issue_ids=[],   #["00100401"],
        num_iids=50,    # Only is read if issue_ids is None or empty, otherwise ignored
        start_date="2014-12-31",
        end_date="2017-12-31",
        lookback_window=252,
        max_windows_per_issue=1000,
        dt= 1.0,        # dt in: X_{t+1} = X_t + mu(X_t) * dt + sigma(X_t) * sqrt(dt) * eps_t
        price_value_col="ClAdjLoc",
        training_data_type="US_Stocks",
        data_source="local",  # "local" or "yahoo",
        dset_path=amgm_config.dataset_root / "data-20250505",
    )

    model_cfg = dict(
        _target_=NeuralSDEMoE,
        lookback_window=dset_cfg["lookback_window"],
        num_features=7,
        hidden_sizes=[32, 16],
        output_size=output_size,
    )
    
    trainer_cfg = dict(
        run_cfg=run_cfg,
        dset_cfg=dset_cfg,
        model_cfg=model_cfg,
        loss_cfg=dict(_target_=nn.MSELoss),
        acc_cfg=dict(_target_=MeanAbsoluteError),
        optim_cfg=dict(_target_=torch.optim.Adam, lr=1e-3, weight_decay=1e-5),
        sched_cfg=None,
        entropy_beta=0.1,   # Regularization strength for entropy loss in MoE model. Set to 0.0 to disable entropy regularization.
        entropy_beta_min=1e-2,
        entropy_beta_warmup_steps=200,
        entropy_beta_decay_steps=800,     # Set to 0 to have a constant entropy_beta value during training
        expert_balance_lambda=0,
    )

    return trainer_cfg
