"""
yolo/layers/route.py
Implementation of YOLOv3 route layer.
"""

from typing import List
import torch
from torch import Tensor

from yolo.darknet import Darknet, DarknetLayer


class Route(DarknetLayer):
    def __init__(self, net: Darknet, index: int, out_channels: int,
                 indexes: List[int]) -> None:
        super().__init__(net, index, out_channels)
        self.indexes = indexes

    def __repr__(self) -> str:
        return f'({self.index}): Route (indexes: {self.indexes})'

    def __str__(self) -> str:
        return repr(self)

    def forward(self, x: Tensor) -> Tensor:
        inputs = [self.net.outputs[i] for i in self.indexes]
        out = torch.cat(inputs, dim=1)
        return out
