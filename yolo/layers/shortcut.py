"""
yolo/layers/shorcut.py
Implementation of YOLOv3 shortcut layer.
"""

from torch import Tensor

from yolo.darknet import Darknet, DarknetLayer


class Shortcut(DarknetLayer):
    def __init__(self, net: Darknet, index: int, out_channels: int,
                 from_: int) -> None:
        super().__init__(net, index, out_channels)
        self.from_ = from_

    def __repr__(self) -> str:
        return f'({self.index}): Shortcut from {self.from_}'

    def __str__(self) -> str:
        return repr(self)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net.outputs[self.from_]
