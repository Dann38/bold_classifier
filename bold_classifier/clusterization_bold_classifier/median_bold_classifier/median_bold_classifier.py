import numpy as np

from ..clusterization_bold_classifier import ClusterizationBoldClassifier


class MedianBoldClassifier(ClusterizationBoldClassifier):
    def evaluation_one_bbox_image(self, image: np.ndarray) -> float:
        base_line_image = self._get_base_line_image(image)
        base_line_image_without_sparces = self._get_rid_spaces(base_line_image)
        if base_line_image_without_sparces.shape[1] == 0:
            return 1
        x = base_line_image_without_sparces.mean(1)
        evaluation = float(np.median(x))
        return evaluation
