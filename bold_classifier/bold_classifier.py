from abc import ABC, abstractmethod
from typing import List

import numpy as np

from dataset_reader.bbox import BBox

from .types_font import PERMISSIBLE_W_BBOX, PERMISSIBLE_H_BBOX, VERTICAL_SPARSE_BETWEEN_CHAR

class BaseBoldClassifier(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray,  bboxes: List[BBox]) -> List[float]:
        pass

    def _get_rid_spaces(self, image: np.ndarray) -> np.ndarray:
        x = image.mean(0)
        not_space = x < VERTICAL_SPARSE_BETWEEN_CHAR
        if len(not_space) > PERMISSIBLE_W_BBOX:
            return image
        return image[:, not_space]

    def _get_base_line_image(self, image: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        if h < PERMISSIBLE_H_BBOX:
            return image
        mean_ = image.mean(1)
        delta_mean = abs(mean_[:-1] - mean_[1:])

        max1 = 0
        max2 = 0
        argmax1 = 0
        argmax2 = 0
        for i, delta_mean_i in enumerate(delta_mean):
            if delta_mean_i <= max2:
                continue
            if delta_mean_i > max1:
                max2 = max1
                argmax2 = argmax1
                max1 = delta_mean_i
                argmax1 = i
            else:
                max2 = delta_mean_i
                argmax2 = i
        h_min = min(argmax1, argmax2)
        h_max = min(max(argmax1, argmax2) + 1, h)
        if h_max-h_min < PERMISSIBLE_H_BBOX:
            return image
        return image[h_min:h_max, :]

    def _is_correct_bbox_image(self, image: np.ndarray) -> bool:
            h, w = image.shape[0:2]
            return h > PERMISSIBLE_H_BBOX and w > PERMISSIBLE_W_BBOX
