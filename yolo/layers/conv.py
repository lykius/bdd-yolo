"""
yolo/layers/conv.py
Implementation of YOLOv3 convolutional layer.
"""

import torch
import torch.nn as nn
from torch import Tensor

from yolo.darknet import Darknet, DarknetLayer


class Conv(DarknetLayer):
    def __init__(self, net: Darknet, index: int, in_channels: int,
                 out_channels: int, kernel_size: int, stride: int,
                 padding: int, activ: nn.Module, batch_norm: bool) -> None:
        super().__init__(net, index, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              bias=(not batch_norm))
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = activ

        self.net.add_module(f'{self.index}_conv', self.conv)
        if self.batch_norm is not None:
            self.net.add_module(f'{self.index}_batch_norm', self.batch_norm)
        if self.activation is not None:
            self.net.add_module(f'{self.index}_activation', self.activation)

    def __repr__(self) -> str:
        s = (f'({self.index}): Convolution (\n'
             f'    {repr(self.conv)}\n')
        if self.batch_norm:
            s += f'    {repr(self.batch_norm)}\n'
        if self.activation:
            s += f'    {repr(self.activation)}\n'
        s += ')'
        return s

    def __str__(self) -> str:
        return f'({self.index}): Convolution'

    def load_weights(self, weights: Tensor) -> Tensor:
        with torch.no_grad():
            parameters = []
            if self.batch_norm is not None:
                parameters.append(self.batch_norm.bias)
                parameters.append(self.batch_norm.weight)
                parameters.append(self.batch_norm.running_mean)
                parameters.append(self.batch_norm.running_var)
            else:
                parameters.append(self.conv.bias)

            for p in parameters:
                p.copy_(weights[:self.out_channels])
                weights = weights[self.out_channels:]

            num_weights = self.conv.weight.numel()
            conv_weights = weights[:num_weights].view_as(self.conv.weight)
            self.conv.weight.copy_(conv_weights)
            weights = weights[num_weights:]

        return weights

    def forward(self, x: Tensor) -> Tensor:
        # if self.index == 0:
        #     with open(str(self.index) + '_input.txt', 'wt') as f:
        #         for c in range(x.size()[1]):
        #             for j in range(x.size()[2]):
        #                 for i in range(x.size()[3]):
        #                     f.write(f'{x[0, c, j, i].item():.4f} ')
        #                 f.write('\n')
        #             f.write('\n')

        x = self.conv(x)

        # if self.index == 0:
        #     with open(str(self.index) + '_conv.txt', 'wt') as f:
        #         for c in range(x.size()[1]):
        #             for j in range(x.size()[2]):
        #                 for i in range(x.size()[3]):
        #                     f.write(f'{x[0, c, j, i].item():.4f} ')
        #                 f.write('\n')
        #             f.write('\n')

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        # if self.index == 0:
        #     with open(str(self.index) + '_batchnorm.txt', 'wt') as f:
        #         for c in range(x.size()[1]):
        #             for j in range(x.size()[2]):
        #                 for i in range(x.size()[3]):
        #                     f.write(f'{x[0, c, j, i].item():.4f} ')
        #                 f.write('\n')
        #             f.write('\n')

        if self.activation is not None:
            x = self.activation(x)

        # if self.index == 0:
        #     with open(str(self.index) + '_activ.txt', 'wt') as f:
        #         for c in range(x.size()[1]):
        #             for j in range(x.size()[2]):
        #                 for i in range(x.size()[3]):
        #                     f.write(f'{x[0, c, j, i].item():.4f} ')
        #                 f.write('\n')
        #             f.write('\n')

        return x
