import numpy as np

from ..clusterization_bold_classifier import ClusterizationBoldClassifier


class PsBoldClassifier(ClusterizationBoldClassifier):
    def evaluation_one_bbox_image(self, image: np.ndarray) -> float:
        base_line_image = self._get_base_line_image(image)  # baseline - main font area
        base_line_image_without_sparces = self._get_rid_spaces(base_line_image)  # removing spaces from a string

        p_img = base_line_image[:, :-1] - base_line_image[:, 1:]
        p_img[abs(p_img) > 0] = 1.
        p_img[p_img < 0] = 0.
        p = p_img.mean()

        s = 1 - base_line_image_without_sparces.mean()

        if p > s or s == 0:
            evaluation = 1.
        else:
            evaluation = p/s
        return evaluation
