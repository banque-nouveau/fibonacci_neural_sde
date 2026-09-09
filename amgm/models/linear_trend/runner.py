import numpy as np
from torch import nn
from lightning import LightningModule
import torch
from amgm.utils import instantiate, normalize_hparams
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torch import autograd

class LinearTrendRunner(LightningModule):

    def __init__(self, run_cfg, dset_cfg, model_cfg, loss_cfg, acc_cfg, optim_cfg, sched_cfg):
        super().__init__()

        # Ensure all configuration types that need to be, are strings for serialization
        run_cfg = normalize_hparams(run_cfg)
        dset_cfg = normalize_hparams(dset_cfg)
        model_cfg = normalize_hparams(model_cfg)
        loss_cfg = normalize_hparams(loss_cfg)
        acc_cfg = normalize_hparams(acc_cfg)
        optim_cfg = normalize_hparams(optim_cfg)
        sched_cfg = normalize_hparams(sched_cfg)

        # Extract lr from optim_cfg if present
        self.lr = None
        if isinstance(optim_cfg, dict):
            self.lr = optim_cfg.get("lr", None)

        # Save lr as a hyperparameter so Lightning lr_find works
        self.save_hyperparameters(ignore=["model", "loss_fn", "acc_fn"])

        self.run_cfg = run_cfg
        self.dset_cfg = dset_cfg  # Kept for logging purposes
        self.model = instantiate(model_cfg)
        self.loss_fn = instantiate(loss_cfg)
        self.acc_fn = instantiate(acc_cfg)

        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.seed = 1

        self.task_type = dset_cfg["task_type"] if "task_type" in dset_cfg else "classification"
        self.threshold = acc_cfg["threshold"] if "threshold" in acc_cfg else 0.5

        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _extract_features_targets(batch):
        return batch.features, batch.targets

    def training_step(self, batch, batch_idx):
        features, targets = self._extract_features_targets(batch)
        if self.task_type == "classification":
            targets = targets.squeeze()     # Because CrossEntropyLoss expect targets to be of shape (batch,) and not (batch,1) 
        pred = self(features)
        loss = self.loss_fn(pred, targets)

        self.train_predictions.append(pred.detach())
        self.train_targets.append(targets)
        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_predictions, dim=0).cpu()
        targets = torch.cat(self.train_targets, dim=0).cpu()
        loss = self.loss_fn(preds, targets)
        train_acc = self.acc_fn(preds, targets)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        self.log("lr", current_lr, prog_bar=True, on_epoch=True)

        self.train_predictions.clear()
        self.train_targets.clear()

    def validation_step(self, batch, batch_idx):
        features, targets = self._extract_features_targets(batch)
        if self.task_type == "classification":
            targets = targets.squeeze()     # Because CrossEntropyLoss expect targets to be of shape (batch,) and not (batch,1)
        pred = self(features)
        self.val_predictions.append(pred)
        self.val_targets.append(targets)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions, dim=0).cpu()
        targets = [t.unsqueeze(0) if t.dim() == 0 else t for t in self.val_targets]
        targets = torch.cat(targets, dim=0).cpu()
        loss = self.loss_fn(preds, targets)
        accuracy = self.acc_fn(preds, targets)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", accuracy, prog_bar=True, on_epoch=True)

        if self.task_type == "classification":
            n = preds.shape[1]
            task = "multiclass" if n > 1 else "binary"
            thr = getattr(self.acc_fn, "threshold", 0.5)
            cm = ConfusionMatrix(task=task, num_classes=n, threshold=thr)(preds, targets).cpu()
            self.val_confusion_matrix = cm

        self.val_predictions.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = instantiate(self.optim_cfg, params=self.parameters())
        scheduler = instantiate(self.sched_cfg, optimizer=optimizer) if self.sched_cfg is not None else None

        cfg = dict(optimizer=optimizer)
        if scheduler is not None:
            cfg["lr_scheduler"] = dict(
                scheduler=scheduler,
                interval="epoch",  # or "step"
                frequency=1,
                monitor=None,  # only needed for schedulers like ReduceLROnPlateau
            )
        return cfg

    def predict_step(self, batch, batch_idx):
        """Builds the prediction outputs for a given batch.

        Args:
            batch (tuple): The input batch containing features, targets, and optional additional data. 

        Returns:
            result (dict): A dictionary containing predicted labels, predicted probabilities, true targets, features, etc.
            result["y_pred"]: Predicted class labels. It can be 0 (NoTouch), 1 (Up), or 2 (Down).
            result["p_pred"]: Predicted probabilities for each class. It is of dimension three, corresponding to the three classes.
        """
        features = batch.features
        targets = batch.targets
            
        outputs = {
            "issue_ids": batch.issue_ids,
            "test_dates": batch.test_dates,
            "slopes": batch.slopes,
            "deviations": batch.deviations,
            "test_prices": batch.test_prices,
            "ci_widths": batch.ci_widths,
            "ci_lowers": batch.ci_lowers,
            "ci_uppers": batch.ci_uppers,
            "margins": batch.margins,
            "price_windows": batch.price_windows,
        }

        logits = self(features)
        
        if self.task_type == "classification":
            num_classes = logits.shape[1]
            if num_classes == 1:
                p_pred = torch.sigmoid(logits)
                y_pred = (p_pred > self.threshold).int()
            else:
                p_pred = torch.softmax(logits, dim=1)
                y_pred = logits.argmax(dim=1)
        else:  # regression
            p_pred = logits 
            y_pred = logits 
            
        result = {
            "y_pred": y_pred.reshape(-1), # reshape to 1D if needed
            "p_pred": p_pred,
            "targets": targets.reshape(-1), # reshape to 1D if needed
            "features": features,
        }
        result.update(outputs)
        return result

    def calculate_and_print_val_metrics(self):
        cf = self.val_confusion_matrix.numpy()
        rate = np.round((cf / cf.sum()), decimals=3)

        accuracy = self.trainer.callback_metrics.get("val/acc", None)
        accuracy = accuracy.item() * 100 if accuracy is not None else float('nan')

        row_sums = cf.sum(axis=1, keepdims=True)  # Sum of each row, shape (3, 1)
        # Divide each element by its row sum
        accuracy_rate = cf / row_sums
        
        print(f"Validation Accuracy: {accuracy:.1f}")
        print(f"Confusion matrix:\n{cf}")
        print(f"Accuracy rates:\n{accuracy_rate}")

        


class SectorIRMLinearTrendRunner(LinearTrendRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.irm_c = 4.0
        self.irm_lambda = 0.1

    def training_step(self, batch, batch_idx):

        features, targets, *_ = batch
        targets = targets.float()
        B, E = features.shape[:2]  # (batch, environment, time, feature channels)
        x = features.reshape(B * E, *features.shape[2:])
        y = targets.reshape(B * E, *targets.shape[2:])
        preds = self(x)

        use_irm = True

        if not use_irm:
            loss = self.loss_fn(preds, y)
            loss = loss.mean()
        else:
            # Evaluate all environments at the same time # shape = (B,E,1)
            preds = preds.reshape(B, E, *targets.shape[2:])

            # Compute losses according to the paper
            # Large Scale Financial Time Series Forecasting with Multi-faceted Model
            # https://dl.acm.org/doi/pdf/10.1145/3604237.3626868

            error = 0.0
            penalty = 0.0

            for env in range(E):
                p = torch.randperm(len(targets))  # shuffle the minibatch samples
                y = targets[p, env]
                y_hat = preds[p, env]

                # # Split the batch in two, and compute the squared gradient "norm" across the parts.
                # # (sec 3.2 in the IRM paper)
                # e1, e2 = errors[0::2].mean(), errors[1::2].mean()
                # g1 = autograd.grad(e1, [w], create_graph=True)[0]
                # g2 = autograd.grad(e2, [w], create_graph=True)[0]
                # penalty += torch.sum((g1**self.irm_c) * (g2**self.irm_c))
                # error += (e1 + e2) / 2

                w = torch.tensor(1.).requires_grad_()
                l = self.loss_fn(w * y_hat, y).mean()
                error += l
                g = autograd.grad(l, [w], create_graph=True)[0]
                penalty += (g**4).sum()

            weight_norm = 0.0
            for w in self.parameters():
                weight_norm += w.norm().pow(2)

            loss = (error / E + 0.001 * penalty / E + 0.001 * weight_norm)

        targets = targets.reshape(*preds.shape)
        self.train_predictions.append(preds)
        self.train_targets.append(targets)

        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_predictions, dim=0).cpu()
        targets = torch.cat(self.train_targets, dim=0).cpu()
        preds = preds.reshape(-1, *preds.shape[2:])
        targets = targets.reshape(-1, *targets.shape[2:])

        loss = self.loss_fn(preds, targets).mean()
        train_acc = self.acc_fn(preds, targets)[0]  # acc_fn returns (accuracy, (p, t))
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        self.log("lr", current_lr, prog_bar=True, on_epoch=True)

        # Clear buffers after logging
        self.train_predictions.clear()
        self.train_targets.clear()


class LinearTrendAccuracy(nn.Module):
    """Computes accuracy for binary classification tasks: Accuracy is the fraction of correct predictions."""

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def predict(self, logits):
        p_pred = torch.sigmoid(logits)
        return (p_pred > self.threshold) * 1

    def forward(self, logits, y_true):
        y_pred = self.predict(logits).float()
        correct_predictions = (y_pred == y_true).float()
        accuracy = correct_predictions.mean()
        return accuracy, (y_pred, y_true)

class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss that applies higher weights to labels with higher Mean Absolute Error.
    
    Args:
        weights: List/array of weights for each label, indexed from -63 to +63.
                 weights[i] corresponds to label value (i - 63).
    """
    def __init__(self, weights):
        super().__init__()
        # Convert to tensor
        weights = torch.tensor(weights, dtype=torch.float32)
        # Normalize weights to have mean of 1.0 (preserves training dynamics)
        weights = weights / weights.mean()
        self.register_buffer('weights', weights)    # Register as buffer so it moves with the model and is saved/loaded with state_dict
        self.num_labels = len(weights)
    
    def forward(self, pred, targets):
        """Compute weighted MSE loss.
        
        Args:
            pred: Model predictions (shape: [batch_size] or [batch_size, 1])
            targets: Target labels from -63 to +63 (shape: [batch_size] or [batch_size, 1])
        
        Returns:
            Weighted mean squared error
        """
        # Ensure pred and targets are 1D
        pred = pred.squeeze()
        targets = targets.squeeze()
        
        # Convert label values to indices: label_value + 63
        target_indices = (targets + 63).long()
        # Clamp to valid range [0, num_labels-1]
        target_indices = torch.clamp(target_indices, 0, self.num_labels - 1)
        
        # Ensure target_indices is on the same device as weights
        target_indices = target_indices.to(self.weights.device)
        
        # Get weights for each sample
        sample_weights = self.weights[target_indices]
        
        # Compute MSE per sample
        mse = (pred - targets) ** 2
        
        # Apply weights and average
        weighted_mse = mse * sample_weights
        return weighted_mse.mean()
