import numpy as np

from ..clusterization_bold_classifier import ClusterizationBoldClassifier


class MeanBoldClassifier(ClusterizationBoldClassifier):
    def evaluation_one_bbox_image(self, image: np.ndarray) -> float:
        base_line_image = self._get_base_line_image(image)
        base_line_image_without_spaces = self._get_rid_spaces(base_line_image)
        if np.isnan(base_line_image_without_spaces).all():
            evaluation = 0.0
        else:
            evaluation = base_line_image_without_spaces.mean()
        return evaluation
