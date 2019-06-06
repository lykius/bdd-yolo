"""
dataset_prep.py
Berkeley Deep Drive dataset preparation.
"""

from pathlib import Path
from zipfile import ZipFile
import shutil
import sys

import bdd.consts as k
import bdd.utils as ut


def dataset_preparation() -> None:
    print('Unzipping archives...')
    zip_files = [('Images', k.IMG_ZIP_FILE), ('Labels', k.LBL_ZIP_FILE)]
    for zip_file in zip_files:
        if not Path(zip_file[1]).exists():
            print(f'{zip_file[0]} zip file not found.')
            exit(-1)
        with ZipFile(zip_file[1]) as z:
            z.extractall()
    print('Done.')

    print('Rearranging directories and files...')
    shutil.rmtree(k.IMG_PATH + '/10k')
    shutil.move(k.IMG_PATH + '/100k/test', k.IMG_PATH)
    shutil.move(k.IMG_PATH + '/100k/train', k.IMG_PATH)
    shutil.move(k.IMG_PATH + '/100k/val', k.IMG_PATH)
    (Path(k.IMG_PATH)/'100k').rmdir()
    (Path(k.ROOT)/'labels').rename(k.JSON_PATH)
    json_path = Path(k.JSON_PATH)
    train_labels_path = json_path/'bdd100k_labels_images_train.json'
    train_labels_path.rename(json_path/'train.json')
    val_labels_path = json_path/'bdd100k_labels_images_val.json'
    val_labels_path.rename(json_path/'val.json')
    print('Done.')

    print('Deleting unlabelled images...')
    t = ut.delete_unlabelled_images(k.TRAIN_IMG_PATH, k.TRAIN_INFO_PATH)
    v = ut.delete_unlabelled_images(k.VAL_IMG_PATH, k.VAL_INFO_PATH)
    print(f'Deleted {t} training images and {v} validation images.')

    print('Resizing images...')
    paths = [('training', k.TRAIN_IMG_PATH),
             ('validation', k.VAL_IMG_PATH),
             ('test', k.TEST_IMG_PATH)]
    for p in paths:
        print(f'---- Resizing {p[0]} images...')
        resized_path = p[1] + '-resized'
        Path(resized_path).mkdir(exist_ok=True)
        n = ut.resize_img_dir(p[1], resized_path)
        shutil.rmtree(p[1])
        shutil.move(resized_path, p[1])
        print(f'---- Resized {n} images for {p[0]}.')
    print('Done.')

    print('Creating list files...')
    tr = ut.create_list_file(k.TRAIN_IMG_PATH, k.TRAIN_LIST_FILE)
    v = ut.create_list_file(k.VAL_IMG_PATH, k.VAL_LIST_FILE)
    t = ut.create_list_file(k.TEST_IMG_PATH, k.TEST_LIST_FILE)
    print(f'Created one file with {tr} training images,'
          f' one file with {v} validation images'
          f' and one file with {t} test images.')

    print('Creating labels files...')
    Path(k.TRAIN_LBL_PATH).mkdir(parents=True, exist_ok=True)
    Path(k.VAL_LBL_PATH).mkdir(parents=True, exist_ok=True)
    t = ut.create_labels(k.TRAIN_INFO_PATH, k.TRAIN_LBL_PATH)
    v = ut.create_labels(k.VAL_INFO_PATH, k.VAL_LBL_PATH)
    print(f'Created {t} files for training and {v} files for validation.')


if __name__ == '__main__':
    try:
        dataset_preparation()
    except:
        print('Something went wrong:', sys.exc_info()[1])
