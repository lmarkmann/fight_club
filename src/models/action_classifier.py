"""Classification of boxing moves using a temporal convolution network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with residual connection and dilated conv."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.bn(self.conv(x)))) + self.residual(x)


class TemporalAttentionPool(nn.Module):
    """Learned attention weights over the time dimension."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time) → transpose for linear
        w = F.softmax(self.attn(x.transpose(1, 2)), dim=1)  # (batch, time, 1)
        return (x * w.transpose(1, 2)).sum(dim=-1)  # (batch, channels)


class BoxingActionClassifier(nn.Module):
    """TCN for boxing action classification.

    Input:  (batch, time, 17, num_coords)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        num_coords: int = 9,
        num_classes: int = 13,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_coords = num_coords
        input_dim = num_keypoints * num_coords

        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            layers.append(
                TemporalConvBlock(
                    in_ch, hidden_channels, kernel_size, dilation=2**i, dropout=dropout
                )
            )
            in_ch = hidden_channels
        self.temporal_conv = nn.Sequential(*layers)
        self.pool = TemporalAttentionPool(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = x.shape
        x = x.view(b, t, -1).permute(0, 2, 1)  # (batch, features, time)
        x = self.temporal_conv(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
