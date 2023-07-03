from ..clasterization_bold_classifier import ClasterizationBoldClassifier
from binarizer.valley_emphasis_binarizer import ValleyEmphasisBinarizer  # TODO Изменить путь
from ..utils import base_line_image, get_rid_spaces
import numpy as np


class HistBoldClassifier(ClasterizationBoldClassifier):
    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        ve_bin = ValleyEmphasisBinarizer()
        return ve_bin.binarize(image)

    def evaluation_method(self, image: np.ndarray) -> float:
        step_hist = 0.05
        img = base_line_image(image)
        img = get_rid_spaces(img)
        if len(img) == 0:
            return 1
        if img.shape[1] == 0:
            return 1
        x = img.mean(1)
        h, b = np.histogram(x, np.arange(0, 1+step_hist, step_hist))
        return b[h.argmax()]
