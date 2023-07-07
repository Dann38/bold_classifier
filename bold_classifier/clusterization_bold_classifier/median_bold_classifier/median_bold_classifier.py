import numpy as np

from ..clusterization_bold_classifier import ClusterizationBoldClassifier


class MedianBoldClassifier(ClusterizationBoldClassifier):
    def evaluation_one_bbox_image(self, image: np.ndarray) -> float:
        step_hist = 0.05
        img = self._get_base_line_image(image)
        img = self._get_rid_spaces(img)
        if img.shape[1] == 0:
            return 1
        x = img.mean(1)
        rez = float(np.median(x))
        return rez
