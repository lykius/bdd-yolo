"""
dataset_test.py
Simple test to view some images from Berkely Deep Drive dataset.
"""

from torch.utils.data.dataloader import DataLoader

from bdd.utils import show_img
import bdd.consts as k
from bdd.dataset import BDDDataset


def main() -> None:
    val_ds = BDDDataset(k.VAL_LIST_FILE, k.VAL_IMG_PATH,
                        k.VAL_LBL_PATH, k.IMG_EXT)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1,
                        collate_fn=BDDDataset.collate_fn)
    iterator = iter(val_dl)

    ax = None
    for _, samples in enumerate(iterator):
        images = samples[0]
        labels = samples[1]
        bboxes = samples[2]
        ax = show_img(images[0, :, :, :], labels[0], bboxes[0],
                      k.IMG_REDUCED_SIZE, ax=ax)
        cmd = input('Press ENTER to continue (type \'q\' to exit) ')
        if cmd == 'q':
            break


if __name__ == '__main__':
    main()
