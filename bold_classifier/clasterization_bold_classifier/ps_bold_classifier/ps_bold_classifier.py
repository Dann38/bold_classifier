import numpy as np

from ..clusterization_bold_classifier import ClasterizationBoldClassifier


class PsBoldClassifier(ClasterizationBoldClassifier):
    def evaluation_method(self, image: np.ndarray) -> float:
        image_p = self._get_base_line_image(image)  # baseline - main font area
        image_s = self._get_rid_spaces(image_p)  # removing spaces from a string
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
