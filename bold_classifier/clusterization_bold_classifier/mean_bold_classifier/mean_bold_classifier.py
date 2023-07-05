import numpy as np

from ..clusterization_bold_classifier import ClusterizationBoldClassifier


class MeanBoldClassifier(ClusterizationBoldClassifier):
    def evaluation_method(self, image: np.ndarray) -> float:
        bl_image = self._get_base_line_image(image)
        image_s = self._get_rid_spaces(bl_image)
        if np.isnan(image_s).all():
            return 0.0
        return image_s.mean()
