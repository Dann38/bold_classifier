import numpy as np

from ..clusterization_bold_classifier import ClasterizationBoldClassifier


class HistBoldClassifier(ClasterizationBoldClassifier):
    def evaluation_method(self, image: np.ndarray) -> float:
        step_hist = 0.05
        img = self._get_base_line_image(image)
        img = self._get_rid_spaces(img)
        if len(img) == 0:
            return 1
        if img.shape[1] == 0:
            return 1
        x = img.mean(1)
        h, b = np.histogram(x, np.arange(0, 1+step_hist, step_hist))
        return b[h.argmax()]
