import math
import torch
from torch import nn


class LinearTrendMLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.mdl = nn.Sequential(
            nn.Linear(input_dim, hidden[0], bias=True),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1], bias=True),
            nn.ReLU(),
            nn.Linear(hidden[1], output_dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mdl(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # x: (B*N, T, D)
        return self.pe[:x.size(1)].unsqueeze(0)  # (1, T, D)


class MyTransformerEncoder(nn.Module):
    """ A custom transformer encoder that applies the same layer multiple times."""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers

    def forward(self, x, src_key_padding_mask=None):
        for _ in range(self.num_layers):
            x = self.layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class LinearTrendTransformer(nn.Module):
    """ Time-series transformer, that operates on the input time-series individually."""
    def __init__(self, input_len, feat_dim, model_dim, nhead, num_layers, output_dim, max_time=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(feat_dim, model_dim, kernel_size=1, padding=0, bias=True)

        self.pos_encoder = SinusoidalPositionalEncoding(model_dim, max_len=max_time)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = MyTransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(model_dim*input_len, output_dim, bias=True)


    def forward(self, x: torch.Tensor, mask: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Time series features, shape (B*N, T, D)
            mask (torch.Tensor): Mask tensor, shape (B*N, T), where 1 indicates valid data
            timestamps (torch.Tensor): Time step IDs, shape (B*N, T), values in [0, max_time_id-1]

        Returns:
            torch.Tensor: Output from the transformer, shape (B*N, T, model_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Reshape to (B, T, D) (with D=1)
        B, T, D = x.shape
        h = self.input_proj(x.transpose(-1, -2)).transpose(-1, -2)  # -> (B, T, model_dim)
        h = h + self.pos_encoder(timestamps)

        # Convert mask to src_key_padding_mask format: True = pad/invalid
        src_key_padding_mask = (mask == 0) if mask is not None else None

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        y = self.classifier(h.reshape((B, -1)))  # -> (B, output_dim)
        y = torch.sigmoid(y)
        return y


class LinearTrendAccuracy(nn.Module):
    """ Computes accuracy for binary classification tasks: Accuracy is the fraction of correct predictions."""

    def forward(self, y_pred, y_true):
        correct_predictions = ((y_pred > 0.5).float() == y_true).float()
        accuracy = correct_predictions.mean().item()
        return accuracy