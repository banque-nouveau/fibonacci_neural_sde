import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassAccuracy

from amgm import config as amgm_config
from amgm.models.mlp import MLPModel

def get_trainer_cfg():
    output_size = 3

    run_cfg = dict(
        run_idx=1,
        rng_seed=1,
        batch_size=200,
        num_workers=0,
        max_epochs=10,
    )

    dset_cfg = dict(
        split_method="equal",
        num_iids=500,   # I tested upto 2000 and didn't get a significant improvement
        start_date="2014-01-01",        # 2014-01-01 for stocks
        end_date_train="2017-12-31",    # 2017-12-31 for stocks
        end_date_test="2020-12-31",     # 2020-12-31 for stocks
        window_size_train=21,  # 252 for 1Y, 378 for 1.5Y, 504 for 2Y
        window_size_test=5,    # 63 for 3M
        task_type="classification",
        label_variable="label_class",
        time_bar="daily",  # daily turns out to be the best
        trend_value_col="ClAdjLoc",
        feature_names=["ClAdjLoc"],
        normalization=dict(ClAdjLoc="sample_minmax"),
        balancing=dict(train="smallest", val="smallest"),
        training_data_type="US_Stocks",
        data_source="local",  # "local" or "yahoo",
        dset_path=amgm_config.dataset_root / "data-20250505",
    )

    model_cfg = dict(
        _target_=MLPModel,
        input_length=dset_cfg["window_size_train"],
        hidden_sizes=[64, 8],  # I test multiple hidden sizes randomly (upto 5000) and found this one to be the best!
        output_size=output_size,
    )

    class_weights = torch.tensor([1.0, 1.0, 1.0])

    trainer_cfg = dict(
        run_cfg=run_cfg,
        dset_cfg=dset_cfg,
        model_cfg=model_cfg,
        loss_cfg=dict(_target_=nn.CrossEntropyLoss, weight=class_weights),
        acc_cfg=dict(_target_=MulticlassAccuracy, num_classes=output_size),
        optim_cfg=dict(_target_=torch.optim.Adam, lr=1e-3, weight_decay=1e-5),
        sched_cfg=dict(_target_=StepLR, step_size=run_cfg["max_epochs"], gamma=0.2),  # I found no decay works better so I set step_size=max_epochs
    )

    return trainer_cfg
