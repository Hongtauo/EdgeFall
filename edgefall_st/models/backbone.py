"""Lightweight 2D feature extractor."""

from __future__ import annotations

from torch import Tensor, nn

from edgefall_st.models.layers import ConvBNAct, InvertedResidual


class LiteBackbone(nn.Module):
    """Produce P3/P4/P5 at strides 8/16/32."""

    def __init__(
        self,
        channels: tuple[int, int, int, int, int] = (16, 32, 64, 128, 192),
        *,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("channels must contain five stage widths")
        c0, c1, c2, c3, c4 = channels
        self.output_channels = {"p3": c2, "p4": c3, "p5": c4}

        self.stem = ConvBNAct(3, c0, 3, 2, activation=activation)
        self.stage1 = nn.Sequential(
            InvertedResidual(c0, c1, stride=2, activation=activation),
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(c1, c2, stride=2, activation=activation),
            InvertedResidual(c2, c2, activation=activation),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(c2, c3, stride=2, activation=activation),
            InvertedResidual(c3, c3, activation=activation),
            InvertedResidual(c3, c3, activation=activation),
        )
        self.stage4 = nn.Sequential(
            InvertedResidual(c3, c4, stride=2, activation=activation),
            InvertedResidual(c4, c4, activation=activation),
        )

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.stem(images)
        features = self.stage1(features)
        p3 = self.stage2(features)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return {"p3": p3, "p4": p4, "p5": p5}
