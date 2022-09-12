import os
import cv2
import torch
import numpy as np
#IMPORT MODEL
from ..models import BiSeNetV2
from ..dataloader import Rs19DatasetConfig


class RailtrackSegmentation:
    def __init__(self, weights, imgsz_h, imgsz_w, model_config, overlay_alpha=0.5):
        if not os.path.isfile(weights):
            raise Exception("{} does not exist".format(weights))

        self._model_config = model_config
        self._overlay_alpha = overlay_alpha
        self._imgsz_h = imgsz_h
        self._imgsz_w = imgsz_w

        self._data_config = Rs19DatasetConfig()
        self._model = BiSeNetV2(n_classes=self._data_config.num_classes)
        self._model.load_state_dict(torch.load(weights)["state_dict"])
        self._model.eval()

        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def run(self, image, only_mask=True):
        orig_height, orig_width = image.shape[:2]
        processed_image = cv2.resize(image, (self._imgsz_h, self._imgsz_w))

        if not only_mask:
            overlay = np.copy(processed_image)

        processed_image = processed_image / 255.0
        processed_image = torch.tensor(processed_image.transpose(2, 0, 1)[np.newaxis, :]).float()

        if torch.cuda.is_available():
            processed_image = processed_image.cuda()

        output = self._model(processed_image)[0]
        mask = (
            torch.argmax(output, axis=0)
            .cpu()
            .numpy()
            .reshape(self._imgsz_h, self._imgsz_w)
        )

        if not only_mask:
            color_mask = np.array(self._data_config.RS19_COLORS)[mask]
            overlay = (((1 - self._overlay_alpha) * overlay) + (self._overlay_alpha * color_mask)).astype("uint8")
            overlay = cv2.resize(overlay, (orig_width, orig_height))
            return mask, overlay

        return mask
