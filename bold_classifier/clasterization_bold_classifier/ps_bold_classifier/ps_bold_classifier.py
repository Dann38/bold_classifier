from ..clasterization_bold_classifier import ClasterizationBoldClassifier
from binarizer.valley_emphasis_binarizer import ValleyEmphasisBinarizer  # TODO Изменить путь
from ..utils import base_line_image, get_rid_spaces
import numpy as np


class PsBoldClassifier(ClasterizationBoldClassifier):
    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        ve_binarizer = ValleyEmphasisBinarizer()
        return ve_binarizer.binarize(image)

    def evaluation_method(self, image: np.ndarray) -> float:
        image_p = base_line_image(image)
        image_s = get_rid_spaces(image_p)
        hw = image_s.shape[0] * image_s.shape[1]
        p_img = image_p[:, :-1] - image_p[:, 1:]
        p_img[p_img > 0] = 1
        p = p_img.sum()
        s = hw - image_s.sum()
        return p / s