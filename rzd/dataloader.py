import tqdm
import os
import numpy as np
import cv2
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, data_path, classes, colors, phase, transform):
        super(BaseDataset, self).__init__()

        assert os.path.isdir(data_path)
        self._data_path = data_path

        self._image_paths = []
        self._gt_paths = []

        self._classes = classes
        self._colors = colors
        self._legend = BaseDataset.show_color_chart(self._classes, self._colors)

        assert phase in ("train", "val", "test")
        self._phase = phase

        self._transform = transform

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image, gt = self._pull_item(idx)

        return image, gt

    def _pull_item(self, idx):
        image = cv2.imread(self._image_paths[idx])
        gt = cv2.imread(self._gt_paths[idx], 0)

        if self._transform is not None:
            image, gt = self._transform(image, gt, self._phase)

        return image, gt

    @property
    def colors(self):
        return self._colors

    @property
    def legend(self):
        return self._legend

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)
