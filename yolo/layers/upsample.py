"""
yolo/layers/upsample.py
Implementation of YOLOv3 upsample layer.
"""

import torch.nn as nn
from torch import Tensor

from yolo.darknet import Darknet, DarknetLayer


class Upsample(DarknetLayer):
    def __init__(self, net: Darknet, index: int, out_channels: int,
                 scale_factor: int, mode: str) -> None:
        super().__init__(net, index, out_channels)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode,
                                    align_corners=True)

        self.net.add_module(f'{self.index}_upsamle', self.upsample)

    def __repr__(self) -> str:
        return f'({self.index}): {repr(self.upsample)}'

    def __str__(self) -> str:
        return (f'({self.index}): Upsample '
                f'(scale_factor: {self.upsample.scale_factor})')

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(x)
