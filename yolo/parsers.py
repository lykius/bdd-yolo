"""
yolo/parsers.py
Parsers for YOLOv3 configuration file.
"""

from typing import Dict, Callable, Optional
import torch.nn as nn

from yolo.darknet import Darknet
from yolo.layers.conv import Conv
from yolo.layers.shortcut import Shortcut
from yolo.layers.route import Route
from yolo.layers.upsample import Upsample
from yolo.layers.detector import Detector


def get_parsers() -> Dict[str, Callable]:
    return {'net': parse_net,
            'convolutional': parse_convolutional,
            'shortcut': parse_shortcut,
            'upsample': parse_upsample,
            'route': parse_route,
            'yolo': parse_yolo}


def get_activation(key: str) -> Optional[nn.Module]:
    if key == 'leaky':
        return nn.LeakyReLU(negative_slope=.1)
    return None


def get_last_layer_out_channels(net: Darknet) -> int:
    if len(net.layers) > 0:
        return net.layers[-1].out_channels
    return net.channels


def parse_net(layer: Dict[str, str]) -> Darknet:
    steps = [int(s.strip()) for s in layer['steps'].split(',')]
    scales = [float(s.strip()) for s in layer['scales'].split(',')]
    steps_scales = list(zip(steps, scales))
    return Darknet(int(layer['batch']), int(layer['subdivisions']),
                   int(layer['max_batches']), int(layer['width']),
                   int(layer['height']), int(layer['channels']),
                   float(layer['learning_rate']), int(layer['burn_in']),
                   steps_scales, float(layer['momentum']),
                   float(layer['decay']), float(layer['angle']),
                   float(layer['saturation']), float(layer['exposure']),
                   float(layer['hue']))


def parse_convolutional(layer: Dict[str, str], net: Darknet) -> Conv:
    layer_index = len(net.layers)
    in_channels = get_last_layer_out_channels(net)
    out_channels = int(layer['filters'])
    kernel_size = int(layer['size'])
    stride = int(layer['stride'])
    padding = 0
    if layer['pad'] == '1':
        padding = kernel_size // 2
    activation = get_activation(layer['activation'])
    batch_norm = layer.get('batch_normalize', '0') == '1'
    return Conv(net, layer_index, in_channels, out_channels, kernel_size,
                stride, padding, activation, batch_norm)


def parse_upsample(layer: Dict[str, str], net: Darknet) -> Upsample:
    layer_index = len(net.layers)
    out_channels = get_last_layer_out_channels(net)
    scale_factor = int(layer['stride'])
    return Upsample(net, layer_index, out_channels, scale_factor, 'bilinear')


def parse_shortcut(layer: Dict[str, str], net: Darknet) -> Shortcut:
    layer_index = len(net.layers)
    out_channels = get_last_layer_out_channels(net)
    from_ = int(layer['from'])
    if from_ < 0:
        from_ = layer_index + from_
    return Shortcut(net, layer_index, out_channels, from_)


def parse_route(layer: Dict[str, str], net: Darknet) -> Route:
    layer_index = len(net.layers)
    indexes = [int(t.strip()) for t in layer['layers'].split(',')]
    for i in range(len(indexes)):
        if indexes[i] < 0:
            indexes[i] = layer_index + indexes[i]
    out_channels = sum(net.layers[i].out_channels for i in indexes)
    return Route(net, layer_index, out_channels, indexes)


def parse_yolo(layer: Dict[str, str], net: Darknet) -> Detector:
    layer_index = len(net.layers)
    net.detector_layers.append(layer_index)
    out_channels = -1
    anc = [int(s.strip()) for s in layer['anchors'].split(',')]
    mask = [int(s.strip()) for s in layer['mask'].split(',')]
    anchors = [(anc[m*2], anc[m*2 + 1]) for m in mask]
    return Detector(net, layer_index, out_channels, int(layer['classes']),
                    anchors, float(layer['jitter']),
                    float(layer['ignore_thresh']),
                    float(layer['truth_thresh']), bool(layer['random']))
