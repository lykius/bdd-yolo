""" bdd/dataset.py
PyTorch Dataset implementation for Berkeley Deep Drive dataset.
"""

import torch
from torch import Tensor
from torchvision.transforms import Compose, ToTensor
from torch.utils.data.dataset import Dataset
from pathlib import Path
from PIL import Image
from typing import Callable, Tuple, List


class BDDDataset(Dataset):
    def __init__(self, listfile: str, img_path: str, lbl_path: str, ext: str,
                 transform: Callable[[Tensor], Tensor] = None) -> None:
        with open(listfile) as f:
            self.ids = f.readlines()
        for i in range(len(self.ids)):
            self.ids[i] = Path(self.ids[i]).stem
        self.img_path = Path(img_path)
        self.lbl_path = Path(lbl_path)
        self.ext = ext
        if transform is not None:
            self.transform = Compose([transform, ToTensor()])
        else:
            self.transform = ToTensor()

    def __repr__(self) -> str:
        return ('BDDDataset\n'
                f'    Images path: {self.img_path}\n'
                f'    Labels path: {self.lbl_path}\n'
                f'    Transform: {self.transform}')

    def __str__(self) -> str:
        return repr(self)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        img_id = self.ids[index]
        img = self.img_path/(img_id + self.ext)
        image = self.transform(Image.open(img))
        labels: List[int] = []
        bboxes: List[List[float]] = []
        lbl = self.lbl_path/(img_id + '.txt')
        with open(lbl) as f:
            lines = f.readlines()
            for l in lines:
                tokens = l[:l.index('\n')].split(' ')
                labels.append(int(tokens[0]))
                x, y = float(tokens[1]), float(tokens[2])
                w, h = float(tokens[3]), float(tokens[4])
                bboxes.append([x, y, w, h])
        item = (image, torch.tensor(labels), torch.tensor(bboxes))
        return item

    @classmethod
    def collate_fn(cls, items: List[Tuple[Tensor, Tensor, Tensor]]) \
            -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        A custom 'collate_fn' must be passed to the DataLoader that
        will use this Dataset, because 'labels' and 'bboxes' have
        different sizes for every image.
        This function receives a list of  N items from the dataset
        (N = size of mini-batch) and must return a single tuple
        where every element is the 'concatenation' of the elements
        of the single items.
        ---
        example
        ---
        N = 3 items per mini-batch
        CxHxW = size of every image
        M = variable number of labels/bboxes per image
        input = [(image_0, labels_0, bboxes_0),
                 (image_1, labels_1, bboxes_1),
                 (image_2, labels_2, bboxes_2)]
        output = ([[image_0],
                   [image_1],
                   [image_2]], -> NxCxHxW tensor
                  [labels_0, labels_1, labels_2], -> list of N 1xM tensors
                  [bboxes_0, bboxes_1, bboxes_2]) -> list of N 4xM tensors
        """
        images: List[Tensor] = []
        labels: List[Tensor] = []
        bboxes: List[Tensor] = []
        for item in items:
            images.append(item[0])
            labels.append(item[1])
            bboxes.append(item[2])
        img = torch.stack(images, dim=0)  # create a NxCxHxW tensor
        return img, labels, bboxes
