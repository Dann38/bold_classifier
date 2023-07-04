import cv2

from ..clasterization_bold_classifier import ClasterizationBoldClassifier
from binarizer.valley_emphasis_binarizer import ValleyEmphasisBinarizer  # TODO Изменить путь

import numpy as np


class PsBoldClassifier(ClasterizationBoldClassifier):
    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        ve_binarizer = ValleyEmphasisBinarizer()  # TODO перенести в ClasterizationBoldClassifier класс как поле
        return ve_binarizer.binarize(image)

    def evaluation_method(self, image: np.ndarray) -> float:
        image_p = self.base_line_image(image)  # baseline - main font area
        image_s = self.get_rid_spaces(image_p)  # removing spaces from a string
        hw = image_s.shape[0] * image_s.shape[1]
        p_img = image_p[:, :-1] - image_p[:, 1:]
        p_img[abs(p_img) > 0] = 1.
        p_img[p_img < 0] = 0.
        p = p_img.sum()
        s = hw - image_s.sum()
        if p > s:
            return 1.
        if s == 0:
            return 1.
        return p / s
