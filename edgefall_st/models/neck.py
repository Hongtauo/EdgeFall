"""Lightweight PAN-FPN neck."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from edgefall_st.models.layers import ConvBNAct, FusionBlock


class LitePANFPN(nn.Module):
    def __init__(
        self,
        input_channels: dict[str, int],
        output_channels: int = 96,
        *,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.lateral = nn.ModuleDict(
            {
                scale: ConvBNAct(
                    input_channels[scale],
                    output_channels,
                    1,
                    activation=activation,
                )
                for scale in ("p3", "p4", "p5")
            }
        )
        self.top_down_p4 = FusionBlock(
            output_channels * 2,
            output_channels,
            activation=activation,
        )
        self.top_down_p3 = FusionBlock(
            output_channels * 2,
            output_channels,
            activation=activation,
        )
        self.down_p3 = ConvBNAct(
            output_channels,
            output_channels,
            3,
            2,
            activation=activation,
        )
        self.bottom_up_p4 = FusionBlock(
            output_channels * 2,
            output_channels,
            activation=activation,
        )
        self.down_p4 = ConvBNAct(
            output_channels,
            output_channels,
            3,
            2,
            activation=activation,
        )
        self.bottom_up_p5 = FusionBlock(
            output_channels * 2,
            output_channels,
            activation=activation,
        )

    def forward(self, features: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        p3 = self.lateral["p3"](features["p3"])
        p4 = self.lateral["p4"](features["p4"])
        p5 = self.lateral["p5"](features["p5"])

        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4_top = self.top_down_p4(torch.cat((p4, p5_up), dim=1))
        p4_up = F.interpolate(p4_top, size=p3.shape[-2:], mode="nearest")
        p3_out = self.top_down_p3(torch.cat((p3, p4_up), dim=1))

        p3_down = self.down_p3(p3_out)
        p4_out = self.bottom_up_p4(torch.cat((p4_top, p3_down), dim=1))
        p4_down = self.down_p4(p4_out)
        p5_out = self.bottom_up_p5(torch.cat((p5, p4_down), dim=1))
        return p3_out, p4_out, p5_out
