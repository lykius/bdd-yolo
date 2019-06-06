"""
yolo/darknet.py
Implementation of YOLOv3 Darknet.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from abc import ABC, abstractmethod


class DarknetLayer(ABC):
    def __init__(self, net: 'Darknet', index: int, out_channels: int) -> None:
        self.net = net
        self.index = index
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def load_weights(self, weights: Tensor) -> Tensor:
        return weights


class Darknet(nn.Module):
    def __init__(self, batch_size: int, subdivisions: int, max_batches: int,
                 width: int, height: int, channels: int, lr: float,
                 burn_in: int, steps: List[Tuple[int, float]], momentum: float,
                 decay: float, angle: float, saturation: float,
                 exposure: float, hue: float) -> None:
        super().__init__()
        self.version = (-1, -1, -1)
        self.already_seen_images = 0
        self.batch_size = batch_size
        self.subdivisions = subdivisions
        self.max_batches = max_batches
        self.width = width
        self.height = height
        self.channels = channels
        self.lr = lr
        self.burn_in = burn_in
        self.steps = steps
        self.momentum = momentum
        self.decay = decay
        self.angle = angle
        self.saturation = saturation
        self.exposure = exposure
        self.hue = hue
        self.layers: List[DarknetLayer] = []
        self.detector_layers: List[int] = []
        self.outputs: List[Tensor] = []

    def __repr__(self) -> str:
        s = ('### Darknet ###\n'
             f'    Batch size: {self.batch_size}\n'
             f'    Subdivisions: {self.subdivisions}\n'
             f'    Max batches: {self.max_batches}\n'
             f'    Width: {self.width}\n'
             f'    Height: {self.height}\n'
             f'    Channels: {self.channels}\n'
             f'    Learning rate: {self.lr}\n'
             f'    Burn in: {self.burn_in}\n'
             f'    Steps: {self.steps}\n'
             f'    Momentum: {self.momentum}\n'
             f'    Decay: {self.decay}\n'
             f'    Angle: {self.angle}\n'
             f'    Saturation: {self.saturation}\n'
             f'    Exposure: {self.exposure}\n'
             f'    Hue: {self.hue}')
        for layer in self.layers:
            s += '\n' + repr(layer)
        return s

    def __str__(self) -> str:
        return ('### Darknet ###\n'
                f'    Number of layers: {len(self.layers)}\n'
                f'    Number of detectors: {len(self.detector_layers)}')

    def load_weights(self, weights_file: str) -> bool:
        with open(weights_file, 'rb') as f:
            version_nums = np.fromfile(f, dtype=np.int32, count=3)
            self.version = (version_nums[0], version_nums[1], version_nums[2])
            self.already_seen_images = np.fromfile(f, dtype=np.uint64, count=1)
            weights = torch.from_numpy(np.fromfile(f, dtype=np.float32))
            for layer in self.layers:
                weights = layer.load_weights(weights)
        return weights.numel() == 0

    def forward(self, x: Tensor) -> Tensor:
        self.outputs.clear()
        detections: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            self.outputs.append(x)
            if i in self.detector_layers:
                detections.append(x)
        return torch.cat(detections, dim=1)
