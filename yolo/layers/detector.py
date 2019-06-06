"""
yolo/detector.py
Implementation of YOLOv3 detector layer.
"""

from typing import List, Tuple
import torch
from torch import Tensor

from yolo.darknet import Darknet, DarknetLayer


class Detector(DarknetLayer):
    def __init__(self, net: Darknet, index: int, out_channels: int,
                 num_classes: int, anchors: List[Tuple[int, int]],
                 jitter: float, ignore_thresh: float, truth_thresh: float,
                 random: bool) -> None:
        super().__init__(net, index, out_channels)
        self.num_classes = num_classes
        self.anchors = anchors
        self.jitter = jitter
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.random = random

    def __repr__(self) -> str:
        return (f'({self.index}): Detector (\n'
                f'    Number of classes: {self.num_classes}\n'
                f'    Anchors: {self.anchors}\n'
                f'    Jitter: {self.jitter}\n'
                f'    Ignore_thresh: {self.ignore_thresh}\n'
                f'    Truth_thresh: {self.truth_thresh}\n'
                f'    Random: {self.random}\n'
                ')')

    def __str__(self) -> str:
        return f'({self.index}): Detector (anchors: {self.anchors})'

    def forward(self, x: Tensor) -> Tensor:
        """
        x is a Tensor of size (B x C x H x W).
        We want to return a Tensor of size
                B x (H * W * num_bboxes) x (4 + 1 + num_classes),
        i.e. a batch of single columns containing predictions for each image.
        In fact, on the C dimension, we can find the
                num_bboxes * (4 + 1 + num_classes)
        values predicted by each cell.
        We stack these values on a single column to be able to
        stack all the predictions made at different scales:
        - every detector is responsible for a specific scale
        - in the forward step of the full network, we collect
          predictions made by every detector, we concatenate them
          and return a single Tensor with all the predictions.
        On every line of the column we want the attributes of a single
        bounding box, to make manipulations more easy.

        After rearranging data, in each line of x we have:
                (tx, ty, tw, th, obj, p0, ..., pC).
        We must transform them according to following equations:
                bx = sigmoid(tx) + cx
                by = sigmoid(ty) + cy
                bw = pw * exp(tw)
                bh = ph * exp(th)
                bobj = sigmoid(obj),
                bp0 = sigmoid(p0), ..., bpC = sigmoid(pC)
        where (bx, by, bw, bh, bobj, bp0, ..., bpC) are the actual
        bounding boxes predictions, (cx, cy) are the predicting cell
        offsets from the top-left corner and (pw, ph) are the
        anchor box sizes.
        """

        # with open(f'{x.size()[3]}_values.txt', 'wt') as f:
        #     for c in range(x.size()[1]):
        #         for j in range(x.size()[2]):
        #             for i in range(x.size()[3]):
        #                 f.write(f'{x[0, c, j, i].item():.4f} ')
        #             f.write('\n')
        #         f.write('\n')

        s = x.size()
        input_width, input_height = s[3], s[2]
        x = x.view(s[0], s[1], s[2] * s[3])
        x = x.transpose(1, 2).contiguous()
        num_cells = x.size()[1]
        num_bboxes = len(self.anchors)
        num_attributes = 4 + 1 + self.num_classes
        x = x.view(-1, num_cells * num_bboxes, num_attributes)

        img_width = self.net.width
        img_height = self.net.height
        stride = img_width // input_width
        cx = torch.linspace(0, img_width - stride, input_width)
        cy = torch.linspace(0, img_height - stride, input_height)
        cxv, cyv = torch.meshgrid(cx, cy)
        cxv = cxv.t().contiguous().view(-1, 1)
        cyv = cyv.t().contiguous().view(-1, 1)
        cx_cy = torch.cat((cxv, cyv), dim=1)
        cx_cy = cx_cy.repeat(1, num_bboxes).view(-1, 2).unsqueeze(0)
        x[:, :, :2] = torch.sigmoid(x[:, :, :2])*stride + cx_cy  # bx, by
        x[:, :, 0] /= self.net.width
        x[:, :, 1] /= self.net.height

        anchors = torch.tensor(self.anchors, dtype=torch.float) \
                       .repeat(num_cells, 1).unsqueeze(0)
        x[:, :, 2:4] = anchors * torch.exp(x[:, :, 2:4])  # bw, bh
        x[:, :, 2] /= self.net.width
        x[:, :, 3] /= self.net.height

        x[:, :, 4:] = torch.sigmoid(x[:, :, 4:])  # bobj, bp0, ..., bpC

        return x
