import cv2

from ..clasterization_bold_classifier import ClasterizationBoldClassifier
from binarizer.valley_emphasis_binarizer import ValleyEmphasisBinarizer  # TODO Изменить путь
from binarizer.adap_binarizer import AdapBinarizer
from ..utils import base_line_image, get_rid_spaces
import numpy as np


class MeanBoldClassifier(ClasterizationBoldClassifier):
    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        ve_bin = ValleyEmphasisBinarizer()
        return ve_bin.binarize(image)

    def evaluation_method(self, image: np.ndarray) -> float:
        bl_image = base_line_image(image)
        image_s = get_rid_spaces(bl_image)
        if np.isnan(image_s).all():
            return 0.0
        return image_s.mean()
