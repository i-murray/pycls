#!/usr/bin/env python3

"""HiroshimaLemon dataset."""

import os
import re

import h5py
import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg


logger = logging.get_logger(__name__)

# Per-channel mean and standard deviation values (in RGB order)
_MEAN = [0.343, 0.287, 0.118]
_STD = [0.358, 0.310, 0.0935]
#_MEAN = [0.485, 0.456, 0.406]
#_STD = [0.229, 0.224, 0.225]

# Constants for lighting normalization on ImageNet (in RGB order)
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua
_EIG_VALS = [[0.2175, 0.0188, 0.0045]]
_EIG_VECS = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]


class HiroshimaLemon(torch.utils.data.Dataset):
    """HiroshimaLemon dataset."""

    def __init__(self, data, split, in_memory=True):
        assert os.path.exists(data), "Data path '{}' not found".format(data)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported".format(split)
        logger.info("Constructing HiroshimaLemon {}...".format(split))
        self._data, self._split= data, split
        self._size = 0
        assert in_memory, "Only in_memory implemented"
        self._in_memory = in_memory
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""

        with h5py.File(self._data,'r') as hf:

            if self._in_memory:
                self._imdb = {}
            if self._split == 'train':
                self._imdb_lookup = {}
                i = 0
                train = hf['train']
                self._class_ids = sorted(f for f in train.keys() if re.match(r"^[0-9]+$", f))
                self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
                for class_id in self._class_ids:
                    cont_id = self._class_id_cont_id[class_id]
                    if self._in_memory:
                        self._imdb[cont_id] = train[class_id][()]
                    size = train[class_id].shape[0]
                    self._imdb_lookup.update(dict(zip(list(range(i,i+size)),list(zip(list(range(0,size)),[cont_id]*size)))))
                    i += size
                self._size = i
                logger.info("Number of images: {}".format(i))
                logger.info("Number of classes: {}".format(len(self._class_ids)))
            else:
                if self._in_memory:
                    self._imdb = hf['test'][()]
                    size = self._imdb.shape[0]
                    logger.info("Number of images: {}".format(size))
                    self._size = size

    def _prepare_im(self, im, transform=True):

        # Train and test setups differ
        train_size, test_size = cfg.TRAIN.IM_SIZE, cfg.TEST.IM_SIZE
        if self._split == "train" and transform:
            # For training use random_sized_crop, horizontal_flip, augment, lighting
            im = transforms.random_sized_crop(im, train_size)
            im = transforms.horizontal_flip(im, 0.5)
            im = transforms.augment(im, cfg.TRAIN.AUGMENT)
            im = transforms.lighting(im, cfg.TRAIN.PCA_STD, _EIG_VALS, _EIG_VECS)
        else:
            # For testing use scale and center crop
            im = transforms.scale_and_center_crop(im, test_size, train_size)
        if transform:
            # For training and testing use color normalization
            im = transforms.color_norm(im, _MEAN, _STD)
            # Convert HWC/RGB/float to CHW/BGR/float format
            im = np.ascontiguousarray(im[:, :, ::-1].transpose([2, 0, 1]))
        return im

    def __getitem__(self, index):

        i, cont_id = self._imdb_lookup[index]
        # Load the image
        im = self._imdb[cont_id][i,:,:,:].astype(np.float32) / 255
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        return im, cont_id

    def display(self, index):

        i, cont_id = self._imdb_lookup[index]
        # Load the image
        im = self._imdb[cont_id][i,:,:,:].astype(np.float32) / 255
        # Prepare the image for training / testing
        im = self._prepare_im(im, False)
        return im, cont_id


    def __len__(self):
        return self._size