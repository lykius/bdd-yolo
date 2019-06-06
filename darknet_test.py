"""
darknet_test.py
Simple program to test YOLOv3 Darknet implementation.
"""

import torch
from torch.utils.data.dataloader import DataLoader

import bdd.utils as bddut
from bdd.dataset import BDDDataset
import bdd.consts as bddk
import yolo.utils as yolout
import yolo.consts as yolok


def main() -> None:
    layers = yolout.read_cfg(yolok.CFG_FILE)
    darknet = yolout.parse_layers(layers)
    darknet.load_weights(yolok.WEIGHTS_FILE)
    darknet.eval()

    val_ds = BDDDataset(bddk.VAL_LIST_FILE, bddk.VAL_IMG_PATH,
                        bddk.VAL_LBL_PATH, bddk.IMG_EXT)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1,
                        collate_fn=BDDDataset.collate_fn)
    iterator = iter(val_dl)

    ax = None
    for _, samples in enumerate(iterator):
        images = samples[0]
        # labels = samples[1]
        # bboxes = samples[2]
        with torch.no_grad():
            predictions = darknet.forward(images)
        result = yolout.process_predictions(
            predictions, yolok.DETECTION_THRESH, yolok.NMS_THRESH)
        r = result[0]

        print('\nPredictions:')
        for i in range(r.size()[0]):
            cl = r[i, 4].int().item()
            score = r[i, 5].item()
            print(f'{bddk.CLASSES[cl]}: {score:.2f}')

        ax = bddut.show_img(images[0, :, :, :], r[:, 4].int(), r[:, :4],
                            bddk.IMG_REDUCED_SIZE, probabilities=r[:, 5],
                            ax=ax)

        cmd = input('\nPress ENTER to continue (type \'q\' to exit) ')
        if cmd == 'q':
            break


if __name__ == '__main__':
    main()
