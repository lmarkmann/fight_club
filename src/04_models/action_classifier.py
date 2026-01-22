"""Temporal convolutional network for boxing action classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """
    A single temporal convolution block with residual connection.

    The dilation parameter controls spacing between filter taps, allowing
    the receptive field to grow exponentially with depth without increasing
    parameter count.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + residual


class BoxingActionClassifier(nn.Module):
    """
    Temporal convolutional network for boxing action classification.

    Input: normalized pose sequence of shape (batch, time, 17, 2)
    Output: class logits of shape (batch, 13)
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        num_coords: int = 2,
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
            out_ch = hidden_channels
            dilation = 2**i  # 1, 2, 4, 8 for receptive field growth
            layers.append(
                TemporalConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.temporal_conv = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time, kp, coords = x.shape

        x = x.view(batch_size, time, -1)
        x = x.permute(0, 2, 1)  # Conv1d expects (batch, channels, time)

        x = self.temporal_conv(x)
        x = x.mean(dim=-1)  # Global average pooling

        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
