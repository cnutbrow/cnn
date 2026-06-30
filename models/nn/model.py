import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class CNNClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []
        in_ch = n_features
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = hidden_dim
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class TinyTransformerClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])


def build_model(arch: str, n_features: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
    if arch == "gru":
        return GRUClassifier(n_features, hidden_dim, num_layers, dropout)
    if arch == "cnn":
        return CNNClassifier(n_features, hidden_dim, num_layers, dropout)
    if arch == "transformer":
        return TinyTransformerClassifier(n_features, hidden_dim, num_layers, dropout)
    raise ValueError(f"Unknown nn.architecture: {arch}")
