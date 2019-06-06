"""
yolo/utils.py
Utilities for YOLOv3 implementation.
"""

from typing import Optional, List, Dict
import torch
from torch import Tensor

from yolo.darknet import Darknet
from yolo.parsers import get_parsers


def read_cfg(filepath: str) -> List[Dict[str, str]]:
    with open(filepath) as f:
        lines = f.readlines()
    layers: List[Dict[str, str]] = []
    layer: Optional[Dict[str, str]] = None
    flines = filter(lambda x: len(x) > 0 and x != '\n' and x[0] != '#', lines)
    for line in flines:
        line = line.replace('\n', '')
        if line.find('[') == 0:
            if layer is not None:
                layers.append(layer)
            layer = {}
            layer['kind'] = line.strip()[1:-1]
        else:
            tokens: List[str] = line.split('=')
            layer[tokens[0].strip()] = tokens[1].strip()
    if layer is not None:
        layers.append(layer)
    return layers


def parse_layers(layers: List[Dict[str, str]]) -> Darknet:
    parsers = get_parsers()

    darknet = parsers['net'](layers[0])

    for layer in layers[1:]:
        parser = parsers[layer['kind']]
        darknet.layers.append(parser(layer, darknet))

    return darknet


def bbox_iou(a: Tensor, b: Tensor) -> float:
    b1, b2 = (a, b) if a[0] <= b[0] else (b, a)
    width_a, height_a = b1[2].item(), b1[3].item()
    left_a, right_a = b1[0].item() - width_a/2, b1[0].item() + width_a/2
    top_a, bottom_a = b1[1].item() - height_a/2, b1[1].item() + height_a/2
    width_b, height_b = b2[2].item(), b2[3].item()
    left_b, right_b = b2[0].item() - width_b/2, b2[0].item() + width_b/2
    top_b, bottom_b = b2[1].item() - height_b/2, b2[1].item() + height_b/2

    if left_b >= right_a or bottom_b <= top_a or top_b >= bottom_a:
        return 0.

    left = left_b
    right = min(right_a, right_b)
    top = max(top_a, top_b)
    bottom = min(bottom_a, bottom_b)
    intersection = (right - left) * (bottom - top)
    union = width_a*height_a + width_b*height_b - intersection

    return intersection / union


def nms_filter(p: Tensor, nms_thresh: float) -> Tensor:
    n = p.size()[0]
    for i in range(n - 1):
        if p[i, 5] > 0.:
            for j in range(i + 1, n):
                if p[j, 5] > 0.:
                    if (bbox_iou(p[i, :4], p[j, :4]) > nms_thresh):
                        p[j, 5] = 0.
    return p[p[:, 5] > 0.]


def process_predictions(predictions: Tensor, thresh: float,
                        nms_thresh: float) -> List[Tensor]:
    # create a list with one tensor of predictions for every image
    batch_size = predictions.size()[0]
    pred = [predictions[i].clone().detach() for i in range(batch_size)]
    result: List[Tensor] = []

    for p in pred:
        # take only predictions with objectness > thresh
        p = p[p[:, 4] > thresh]

        if p.numel() > 0:
            # for every predicted object, take only the class with
            # the max score 
            class_scores_max = torch.max(p[:, 5:], 1)
            class_scores = torch.stack((class_scores_max.indices.float(),
                                        class_scores_max.values), dim=1)
            # the final class score is computed as:
            #           objectness * class_score
            class_scores[:, 1] = p[:, 4] * class_scores[:, 1]
            p = torch.cat((p[:, :4], class_scores), dim=1)
            # take only predictions with class score > thresh
            p = p[p[:, 5] > thresh]

            if p.numel() > 0:
                classes = set(p[i, 4].int().item() for i in range(p.size()[0]))
                r = []
                # for every class (among the predicted ones),
                # apply non-maximum suppression to filter multiple
                # predictions for the same object
                for cl in classes:
                    pcl = p[p[:, 4].int() == cl]
                    _, indices = pcl[:, 5].sort(dim=0, descending=True)
                    pcl_sorted = pcl[indices, :]

                    r.append(nms_filter(pcl_sorted, nms_thresh))

                p = torch.cat(r, dim=0)
                # sort predictions from the left to the right
                _, indices = p[:, 0].sort(dim=0)
                p = p[indices, :]

        result.append(p)

    return result
