"""
bdd/utils.py
Utilities for Berkeley Deep Drive dataset.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from matplotlib.axes import Axes
from torch import Tensor
from torchvision.transforms import Resize
from PIL import Image
from typing import Tuple, Generator, Optional

import bdd.consts as k


def delete_unlabelled_images(img_path: str, info_path: str) -> int:
    images = set(p.name for p in Path(img_path).glob('*' + k.IMG_EXT))
    with open(info_path) as f:
        info = json.load(f)
    labelled_images = set(entry['name'] for entry in info)
    unlabelled_images = images - labelled_images
    for u in unlabelled_images:
        (Path(img_path)/u).unlink()
    return len(unlabelled_images)


def create_labels(info_path: str, labels_path: str) -> int:
    dx = 1./k.IMG_ORIGINAL_SIZE[0]
    dy = 1./k.IMG_ORIGINAL_SIZE[1]
    count = 0
    with open(info_path) as f:
        info = json.load(f)
    for entry in info:
        img_id = Path(entry['name']).stem
        filename = img_id + '.txt'
        with open(Path(labels_path)/filename, 'wt') as f:
            labels = entry['labels']
            for label in labels:
                if 'box2d' in label:
                    cls_id = k.CLASSES.index(label['category'])
                    bbox = label['box2d']
                    x1, y1 = bbox['x1'], bbox['y1']
                    x2, y2 = bbox['x2'], bbox['y2']
                    x = ((x1 + x2) / 2) * dx
                    y = ((y1 + y2) / 2) * dy
                    w, h = (x2 - x1) * dx, (y2 - y1) * dy
                    f.write(f'{cls_id} {x} {y} {w} {h}\n')
            count += 1
    return count


def create_list_file(images_path: str, listfile: str) -> int:
    count = 0
    with open(listfile, 'wt') as f:
        for p in Path(images_path).glob('*' + k.IMG_EXT):
            f.write(str(p.absolute()) + '\n')
            count += 1
    return count


def resize_img_dir(input_path: str, output_path: str) -> int:
    resize = Resize((k.IMG_REDUCED_SIZE[1], k.IMG_REDUCED_SIZE[0]))
    count = 0
    total = len(list(Path(input_path).glob('*' + k.IMG_EXT)))
    for img_path in Path(input_path).glob('*' + k.IMG_EXT):
        input_img = Image.open(img_path)
        output_img = resize(input_img)
        output_img.save(Path(output_path)/img_path.name)
        count += 1
        if count >= 1000 and count % 1000 == 0:
            print(f'-------- {count}/{total}')
    return count


def colors_generator() -> Generator:
    colors = ['r', 'y', 'g', 'c', 'm', 'b']
    i = 0
    while True:
        yield colors[i]
        i = 0 if i == (len(colors) - 1) else i + 1


def draw_outline(patch: patches.Patch, lw: int) -> None:
    stroke = patheffects.Stroke(linewidth=lw, foreground='black')
    normal = patheffects.Normal()
    patch.set_path_effects([stroke, normal])


def show_img(img: Tensor, labels: Tensor, bboxes: Tensor,
             img_size: Tuple[int, int], **kwargs) -> None:
    ax: Optional[Axes] = kwargs.get('ax')
    probabilities: Optional[Tensor] = kwargs.get('probabilities')
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        # fig.suptitle('Berkeley Deep Drive')
        ax.axis('off')
        ax.set(title='')
    else:
        ax.clear()
    ax.imshow(img.numpy().transpose((1, 2, 0)))
    colors = colors_generator()
    for i in range(labels.size()[0]):
        c = next(colors)
        x, y, w, h = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
        top = (y - h/2) * img_size[1]
        left = (x - w/2) * img_size[0]
        width = w * img_size[0]
        height = h * img_size[1]
        rect = patches.Rectangle((left, top), width, height,
                                 color=c, linewidth=2, fill=False)
        ax.add_patch(rect)
        draw_outline(rect, 6)
        label = k.CLASSES[labels[i].item()]
        if probabilities is not None:
            label += f' {probabilities[i].item():.2f}'

        text = ax.text(left, top, label, horizontalalignment='left',
                       verticalalignment='top', color=c)
        draw_outline(text, 3)
    plt.show(block=False)
    return ax
