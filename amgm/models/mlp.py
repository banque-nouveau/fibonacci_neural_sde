from torch import nn
import torch
import torch.nn.functional as F

class MLPModel(nn.Module):

    def __init__(self, input_length, feats=1, hidden_sizes=(64, 8), output_size=1):
        super().__init__()

        self.classifier = nn.Sequential(nn.Linear(input_length * feats, hidden_sizes[0]), nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            self.classifier += nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU())
        self.classifier += nn.Sequential(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        y = self.classifier(x)
        return y

class MLPModel_time(nn.Module):

    def __init__(self, input_length, feats=1, hidden_sizes=(64, 8), output_size=1, output_range=63):
        super().__init__()
        self.output_range = output_range

        self.classifier = nn.Sequential(nn.Linear(input_length * feats, hidden_sizes[0]), nn.Tanh())
        for i in range(len(hidden_sizes) - 1):
            self.classifier += nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.Tanh())
        self.classifier += nn.Sequential(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        y = self.classifier(x)
        y = self.output_range * torch.tanh(y)  # Scale the output to be between -output_range and output_range
        return y
    
class MLPModel_residual(nn.Module):

    def __init__(self, input_length, feats=1, hidden_sizes=(64, 8), output_size=1, output_range=50):
        super().__init__()
        self.output_range = output_range

        self.classifier = nn.Sequential(nn.Linear(input_length * feats, hidden_sizes[0]), 
                                         nn.Tanh())
        for i in range(len(hidden_sizes) - 1):
            self.classifier += nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), 
                                              nn.Tanh()
                                             )
        self.classifier += nn.Sequential(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        y = self.classifier(x)
        y = self.output_range * torch.tanh(y)
        return y
    
class MLPMultiChannel(nn.Module):
    def __init__(self, input_length, feats=1, hidden_sizes=(64, 8), output_size=1):
        super().__init__()
        if hidden_sizes[0] % feats != 0:
            raise ValueError(f"hidden_sizes[0] ({hidden_sizes[0]}) is not divisible by the number of feats ({feats})")
        
        # Create separate linear layers for each feature channel for the first layer
        self.first_layers = nn.ModuleList([
            nn.Linear(input_length, hidden_sizes[0] // feats) for _ in range(feats)
        ])

        # Subsequent layers after concatenation
        layers = []
        input_size = hidden_sizes[0]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(input_size, hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            input_size = hidden_sizes[i + 1]
        self.classifier = nn.Sequential(*layers)

        # Final output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # x shape: (batch, input_length, feats)
        # Apply separate linear transformations to each feature channel
        feature_outputs = []
        for i, layer in enumerate(self.first_layers):
            feature_input = x[:, :, i]              # (batch, input_length) for each feature
            feature_out = layer(feature_input)      # (batch, hidden_sizes[0]//feats)
            feature_outputs.append(feature_out)

        # Concatenate outputs from all features
        x = torch.cat(feature_outputs, dim=1)       # (batch, hidden_sizes[0])
        x = self.classifier(x)
        return self.output_layer(x)


class NeuralSDEMLP(nn.Module):
    """Simple MLP for neural SDE runner.

    Expects:
        x_hist: (batch, lookback_window)
        f_t: (batch, num_features)
    Returns:
        mu, sigma: both (batch, output_size)
    """

    def __init__(self, lookback_window, num_features, hidden_sizes=(64, 32), output_size=1, gate_temperature=1.0):
        super().__init__()

        input_dim = lookback_window + num_features
        layers = [nn.Linear(input_dim, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU()]

        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_sizes[-1], output_size)
        self.sigma_head = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x_hist, f_t):
        x = torch.cat([x_hist, f_t], dim=1)
        h = self.backbone(x)
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h))
        return mu, sigma, torch.zeros((mu.shape[0], 3), device=mu.device, dtype=mu.dtype)


class ExpertHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # SiLU / Swish works better than ReLU for smooth drift/diffusion prediction
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class GateHead(nn.Module):
    """
    Powerful feedforward gating block with SwiGLU non-linearities,
    LayerNorm, and residual connections.
    """
    def __init__(self, in_dim, hidden_dim=64, num_experts=3):
        super().__init__()
        # Gate & Value projections for SwiGLU
        self.fc_gate = nn.Linear(in_dim, hidden_dim)
        self.fc_value = nn.Linear(in_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Final projection to logits
        self.out_proj = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # SwiGLU interaction: Swish(x W_g) * (x W_v)
        h = F.silu(self.fc_gate(x)) * self.fc_value(x)
        h = self.norm1(h)
        
        # Residual non-linear transformation
        h = h + F.silu(self.fc_out(h))
        h = self.norm2(h)
        
        return self.out_proj(h)
    
class NeuralSDEMoE(nn.Module):
    """
    MoE Neural SDE with explicit structural constraints for:
      - Bounce Expert (Elastic restoring force relative to closest Fib level)
      - Break Expert (Momentum continuation)
      - Hover Expert (Bounded drift, low volatility)
    """

    def __init__(self, lookback_window, num_features=7, hidden_sizes=(64, 32), output_size=1, gate_temperature=1.0):
        super().__init__()
        self.gate_temperature = gate_temperature

        self.lookback_window = lookback_window
        self.num_features = num_features

        # Shared or independent backbone feature extractor
        layers = [nn.Linear(lookback_window, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU()]
        
        self.encoder = nn.Sequential(*layers)
        h_dim = hidden_sizes[-1] + num_features

        # 1. Gating Network (Outputs 3 logits)
        self.gate_head = GateHead(h_dim, hidden_dim=64, num_experts=3)

        # 2. Expert Heads for Drift Magnitude
        self.mu_bounce_head = ExpertHead(h_dim, hidden_dim=32, out_dim=output_size)
        self.mu_break_head  = ExpertHead(h_dim, hidden_dim=32, out_dim=output_size)
        self.mu_hover_head  = ExpertHead(h_dim, hidden_dim=32, out_dim=output_size)

        # 3. Expert Heads for Diffusion
        self.sigma_bounce_head = ExpertHead(h_dim, hidden_dim=32, out_dim=output_size)
        self.sigma_break_head  = ExpertHead(h_dim, hidden_dim=32, out_dim=output_size)
        self.sigma_hover_head  = ExpertHead(h_dim, hidden_dim=32, out_dim=output_size)

    def forward(self, x_hist, f_t):
        """
        x_hist: (batch, lookback_window)
        f_t: (batch, 7) -> assumed signed distances: f_t_k = X_t - L_k
                           (If f_t are absolute distances, pass sign separately)
        """
        # -------------------------------------------------------------
        # Step A: Identify Closest Fib Level & Direction for Bounce
        # -------------------------------------------------------------
        # Find index of minimum absolute distance to Fib levels
        abs_distances = torch.abs(f_t)  # (batch, 7)
        closest_idx = torch.argmin(abs_distances, dim=1, keepdim=True)  # (batch, 1)
        closest_signed_dist = torch.gather(f_t, 1, closest_idx)  # (batch, 1)

        # Direction of bounce:
        # If X_t > L_k (signed_dist > 0), bounce pushes UP (+1)
        # If X_t < L_k (signed_dist < 0), bounce pushes DOWN (-1)
        bounce_sign = torch.sign(closest_signed_dist)
        bounce_sign = torch.where(bounce_sign == 0, torch.ones_like(bounce_sign), bounce_sign)  # Zero handling: treat as +1 (upward bounce)

        # -------------------------------------------------------------
        # Step B: Identify Recent Momentum Direction for Break
        # -------------------------------------------------------------
        # Recent velocity from the end of lookback window: X_t - X_{t-1}
        recent_velocity = x_hist[:, -1:] - x_hist[:, -2:-1]
        break_sign = torch.sign(recent_velocity)
        break_sign = torch.where(break_sign == 0, torch.ones_like(break_sign), break_sign)  #Zero handling: treat as +1 (upward break)

        # -------------------------------------------------------------
        # Step C: Encoder Feature Extraction & Gating
        # -------------------------------------------------------------
        # 1. Temporal encoder processes ONLY history
        Z_t = self.encoder(x_hist)  # Shape: (batch_size, hidden_dim)
        
        # 2. Concatenate latent path vector with instantaneous normalized features
        h = torch.cat([Z_t, f_t], dim=1)  # Shape: (batch_size, hidden_dim + feature_dim)

        # 3. Pass concatenated vector 'h' to gating head and expert heads
        pi = F.softmax(self.gate_head(h) / self.gate_temperature, dim=-1)
        pi_bounce = pi[:, 0:1]
        pi_break  = pi[:, 1:2]
        pi_hover  = pi[:, 2:3]

        # -------------------------------------------------------------
        # Step D: Expert Drift Outputs
        # -------------------------------------------------------------
        mu_bounce = bounce_sign * F.softplus(self.mu_bounce_head(h))    # Bounce Expert: Direction * positive
        mu_break  = break_sign * F.softplus(self.mu_break_head(h))      # Break Expert: Momentum direction * positive     
        mu_hover  = 0.05 * torch.tanh(self.mu_hover_head(h))            # Hover Expert: Strongly damped drift near zero

        # MoE Drift: \mu = \sum \pi_i * \mu_i
        mu_out = pi_bounce * mu_bounce + pi_break * mu_break + pi_hover * mu_hover

        # -------------------------------------------------------------
        # Step E: Expert Diffusion Outputs
        # -------------------------------------------------------------
        sigma_bounce = F.softplus(self.sigma_bounce_head(h)) + 1e-3
        sigma_break  = F.softplus(self.sigma_break_head(h)) + 1e-3
        sigma_hover  = 0.001 + 0.02 * torch.sigmoid(self.sigma_hover_head(h))   # Hover diffusion: volatility compression

        # MoE Variance: \sigma^2 = \sum \pi_i * \sigma_i^2
        var_out = (pi_bounce * (sigma_bounce**2) + pi_break * (sigma_break**2) + pi_hover * (sigma_hover**2))
        sigma_out = torch.sqrt(var_out)

        return mu_out, sigma_out, pi

class PatchNeuralSDETransformer(nn.Module):
    """Patch-based Transformer: Downsamples sequence length before attention."""
    def __init__(self, lookback_window=252, num_features=7, patch_size=6, d_model=32, output_size=1, hidden_sizes=[32, 16]):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = lookback_window // patch_size  # 252 / 6 = 42 tokens
        
        # Maps 1 patch (6 daily prices) -> d_model
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # Shallow 1-layer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.fc_head = nn.Sequential(
            nn.Linear(d_model + num_features, 32),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(32, output_size)
        self.sigma_head = nn.Linear(32, output_size)

    def forward(self, x_hist, f_t):
        # x_hist: (batch, 252) -> reshape to patches: (batch, 42, 6)
        b = x_hist.shape[0]
        x_patched = x_hist.view(b, self.num_patches, self.patch_size)
        
        # Project patches + add spatial encoding
        x_emb = self.patch_embed(x_patched) + self.pos_encoder  # (batch, 42, d_model)
        
        # Attention over 42 tokens instead of 252 days
        h_seq = self.transformer(x_emb).mean(dim=1)
        
        # Fuse with Fibonacci features
        fused = self.fc_head(torch.cat([h_seq, f_t], dim=-1))
        
        mu = self.mu_head(fused)
        sigma = F.softplus(self.sigma_head(fused))
        return mu, sigma, torch.zeros((b, 3), device=mu.device, dtype=mu.dtype)