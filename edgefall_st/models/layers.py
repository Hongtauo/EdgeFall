"""NPU-friendly convolutional building blocks."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn


def _activation(name: str) -> Callable[[], nn.Module]:
    if name == "relu":
        return lambda: nn.ReLU(inplace=True)
    if name == "silu":
        return lambda: nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        *,
        groups: int = 1,
        activation: str | None = "silu",
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation is not None:
            layers.append(_activation(activation)())
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNet-style block using only standard 2D operators."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        expansion: float = 2.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden_channels = max(out_channels, int(round(in_channels * expansion)))
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct(
                in_channels,
                hidden_channels,
                1,
                activation=activation,
            ),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                3,
                stride,
                groups=hidden_channels,
                activation=activation,
            ),
            ConvBNAct(
                hidden_channels,
                out_channels,
                1,
                activation=None,
            ),
        )
        self.output_activation = _activation(activation)()

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.block(inputs)
        if self.use_residual:
            outputs = outputs + inputs
        return self.output_activation(outputs)


class FusionBlock(nn.Module):
    """Depthwise fusion block used in the PAN-FPN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.project = ConvBNAct(
            in_channels,
            out_channels,
            1,
            activation=activation,
        )
        self.spatial = InvertedResidual(
            out_channels,
            out_channels,
            expansion=2.0,
            activation=activation,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.spatial(self.project(inputs))
