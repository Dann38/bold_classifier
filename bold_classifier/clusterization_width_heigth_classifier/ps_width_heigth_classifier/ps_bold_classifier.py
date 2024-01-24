import numpy as np

from ..clusterization_width_heigth_classifier import ClusterizationWidthHeigthClassifier
from typing import Tuple

class PsWidthHeigthClassifier(ClusterizationWidthHeigthClassifier):
    def evaluation_one_bbox_image(self, image: np.ndarray) -> Tuple[float, float]:
        base_line_image = self._get_base_line_image(image)  # baseline - main font area
        s_img = 1 - self._get_rid_spaces(base_line_image)  # removing spaces from a string

        p_img = base_line_image[:, :-1] - base_line_image[:, 1:]
        p_img[abs(p_img) > 0] = 1.
        p_img[p_img < 0] = 0.

        p = p_img.sum()
        s = s_img.sum()
        # return (base_line_image_without_sparces.mean(), p_img.mean())
        if p == 0 or s==0:
            evaluation1 = 0.
            evaluation2 = 1.
        else:
            h = base_line_image.shape[0]
            evaluation1 = s/(p*h)
            evaluation2 = p_img.mean()/s_img.mean()
        return (evaluation1, evaluation2)
