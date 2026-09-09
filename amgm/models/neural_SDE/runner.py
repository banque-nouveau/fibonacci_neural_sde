from lightning import LightningModule
import torch
from amgm.utils import instantiate, normalize_hparams
import torch.nn.functional as F
import math


class EntropyBetaScheduler:
    """Warmup + cosine decay scheduler for entropy regularization strength."""

    def __init__(self, beta_max: float, beta_min: float, warmup_steps: int, decay_steps: int):
        self.beta_max = float(beta_max)
        self.beta_min = float(beta_min)
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)
        self.current_step = 0

    def step(self) -> float:
        if self.current_step < self.warmup_steps:
            beta = self.beta_max
        elif self.current_step >= (self.warmup_steps + self.decay_steps):
            beta = self.beta_min
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.decay_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            beta = self.beta_min + (self.beta_max - self.beta_min) * cosine_decay

        self.current_step += 1
        return float(beta)

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = int(state_dict.get("current_step", 0))

class NeuralSDERunner(LightningModule):

    def __init__(
        self,
        run_cfg,
        dset_cfg,
        model_cfg,
        loss_cfg,
        acc_cfg,
        optim_cfg,
        sched_cfg,
        entropy_beta=0.1,
        entropy_beta_min=1e-3,
        entropy_beta_warmup_steps=200,
        entropy_beta_decay_steps=1000,
        expert_balance_lambda=0.1,
        compile_model=True,
    ):
        super().__init__()

        # Ensure all configuration types that need to be, are strings for serialization
        run_cfg = normalize_hparams(run_cfg)
        dset_cfg = normalize_hparams(dset_cfg)
        model_cfg = normalize_hparams(model_cfg)
        loss_cfg = normalize_hparams(loss_cfg)
        _ = normalize_hparams(acc_cfg)
        optim_cfg = normalize_hparams(optim_cfg)
        sched_cfg = normalize_hparams(sched_cfg)

        # Extract lr from optim_cfg if present
        self.lr = None
        if isinstance(optim_cfg, dict):
            self.lr = optim_cfg.get("lr", None)

        self.entropy_beta = float(entropy_beta)
        self.entropy_beta_min = float(entropy_beta if entropy_beta_min is None else entropy_beta_min)
        self.entropy_beta_warmup_steps = int(entropy_beta_warmup_steps)
        self.entropy_beta_decay_steps = int(entropy_beta_decay_steps)
        self.expert_balance_lambda = float(expert_balance_lambda)
        self.entropy_beta_scheduler = None
        if self.entropy_beta_decay_steps > 0:
            self.entropy_beta_scheduler = EntropyBetaScheduler(
                beta_max=self.entropy_beta,
                beta_min=self.entropy_beta_min,
                warmup_steps=self.entropy_beta_warmup_steps,
                decay_steps=self.entropy_beta_decay_steps,
            )
        
        # Save lr as a hyperparameter so Lightning lr_find works.
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        self.run_cfg = run_cfg
        self.dset_cfg = dset_cfg
        self.model = instantiate(model_cfg)
        self.loss_fn = instantiate(loss_cfg)

        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.eps = float(dset_cfg.get("eps", 1e-6)) if isinstance(dset_cfg, dict) else 1e-6
        self.default_dt = float(dset_cfg.get("dt", 1.0)) if isinstance(dset_cfg, dict) else 1.0
        
        if compile_model:
            self.model = torch.compile(self.model)

    def forward(self, x_window, f_t):
        return self.model(x_window, f_t)  # type: ignore[operator]

    def _step_nll(self, x_t, x_tp1, mu, sigma):
        sigma = sigma + self.eps        # eps for numerical stability
        dt = torch.full_like(x_tp1, self.default_dt)
        dx = x_tp1 - x_t
        var = (sigma ** 2) * dt + self.eps
        mean = mu * dt

        nll = 0.5 * (torch.log(var) + ((dx - mean) ** 2) / var)
        return nll.mean()

    def _prepare_batch(self, batch):
        x_window = batch.price_window   # x_window = X[t-w, ..., t]
        x_t = x_window[:, -1:]          # x_t = X[t]
        x_tp1 = batch.nxt_price         # x_tp1 = X[t+1]
        f_t = batch.features         
        fib_levels = batch.fib_levels

        return x_window, f_t, x_t, x_tp1, fib_levels

    def _compute_loss(self, batch, entropy_beta=None):
        x_window, f_t, x_t, x_tp1, fib_levels = self._prepare_batch(batch)
        mu, sigma, pi = self.forward(x_window, f_t)
        sde_loss = self._step_nll(x_t, x_tp1, mu, sigma)
        
        # Calculate Categorical Entropy per sample: H(pi) = - \sum pi_i * log(pi_i + eps)
        entropy_per_sample = -torch.sum(pi * torch.log(pi + 1e-8), dim=-1) # shape: (batch_size,)
        mean_entropy = torch.mean(entropy_per_sample)

        # pi shape: (batch_size, 3)
        f_m = torch.mean(pi, dim=0)  # Average utilization of each expert across the batch
        # balance_loss = 3.0 * torch.sum(f_m ** 2)  # Minimum is 1.0 when f_m = [1/3, 1/3, 1/3]
        # CV Loss: 0 when f_m = [1/3, 1/3, 1/3], scales gracefully as load unbalances
        balance_loss = torch.sum((f_m - (1.0 / 3.0)) ** 2)
        pi_var_per_expert = torch.var(pi, dim=0, unbiased=False)
        mean_pi_var = torch.mean(pi_var_per_expert)
        # Total Loss: Penalize low entropy to prevent premature expert collapse
        beta = self.entropy_beta if entropy_beta is None else float(entropy_beta)
        entropy_term = beta * mean_entropy
        balance_term = self.expert_balance_lambda * balance_loss
        
        loss = sde_loss - entropy_term + balance_term
        
        x_tp1_pred = x_t + mu * self.default_dt     # mean prediction for x_tp1

        return {
            "loss": loss,
            "sde_loss": sde_loss,
            "entropy_term": entropy_term,
            "balance_term": balance_term,
            "mean_entropy": mean_entropy,
            "balance_loss": balance_loss,
            "mean_pi_var": mean_pi_var,
            "pi_mean": f_m,
            "pi_var": pi_var_per_expert,
            "beta": beta,
            "x_window": x_window,
            "features": f_t,
            "x_t": x_t,
            "x_tp1": x_tp1,
            "x_tp1_pred": x_tp1_pred,
            "fib_levels": fib_levels,
            "mu": mu,
            "sigma": sigma,
        }

    def training_step(self, batch, batch_idx):
        if self.entropy_beta_scheduler is not None:
            self.entropy_beta = self.entropy_beta_scheduler.step()

        out = self._compute_loss(batch, entropy_beta=self.entropy_beta)
        loss = out["loss"]
        sigma_pos = out["sigma"].mean()
        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/entropy_beta", self.entropy_beta, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/loss_sde", out["sde_loss"], prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss_entropy_term", out["entropy_term"], prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss_balance_term", out["balance_term"], prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/mean_pi_var", out["mean_pi_var"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/pi_min_mean", out["pi_mean"].min(), prog_bar=False, on_step=False, on_epoch=True)
        for idx in range(out["pi_mean"].shape[0]):
            self.log(f"train/pi_mean_e{idx}", out["pi_mean"][idx], prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"train/pi_var_e{idx}", out["pi_var"][idx], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/drift_abs", out["mu"].abs().mean(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/diffusion", sigma_pos, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self._compute_loss(batch)
        loss = out["loss"]
        rmse = torch.sqrt(F.mse_loss(out["x_tp1_pred"], out["x_tp1"]))
        sigma_pos = out["sigma"].mean()
        
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/rmse_next", rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/entropy_beta", self.entropy_beta, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/loss_sde", out["sde_loss"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/loss_entropy_term", out["entropy_term"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/loss_balance_term", out["balance_term"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/mean_pi_var", out["mean_pi_var"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/pi_min_mean", out["pi_mean"].min(), prog_bar=False, on_step=False, on_epoch=True)
        for idx in range(out["pi_mean"].shape[0]):
            self.log(f"val/pi_mean_e{idx}", out["pi_mean"][idx], prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"val/pi_var_e{idx}", out["pi_var"][idx], prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/drift_abs", out["mu"].abs().mean(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/diffusion", sigma_pos, prog_bar=False, on_step=False, on_epoch=True)

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
        out = self._compute_loss(batch)
        sigma = out["sigma"] + self.eps

        result = {
            "test_dates": batch.test_dates,
            "x_window": out["x_window"].detach().cpu(),
            "features": out["features"].detach().cpu(),
            "x_t": out["x_t"].detach().cpu(),
            "x_tp1": out["x_tp1"].detach().cpu(),
            "x_tp1_pred": out["x_tp1_pred"].detach().cpu(),
            "fib_levels": out["fib_levels"].detach().cpu(),
            "drift": out["mu"].detach().cpu(),
            "diffusion": sigma.detach().cpu(),
            "batch_loss": out["loss"].detach().cpu(),
        }
        return result

    def on_save_checkpoint(self, checkpoint):
        if self.entropy_beta_scheduler is not None:
            checkpoint["entropy_beta_scheduler"] = self.entropy_beta_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.entropy_beta_scheduler is not None and "entropy_beta_scheduler" in checkpoint:
            self.entropy_beta_scheduler.load_state_dict(checkpoint["entropy_beta_scheduler"])
    