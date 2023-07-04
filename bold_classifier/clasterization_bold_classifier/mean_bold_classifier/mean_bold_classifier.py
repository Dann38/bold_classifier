import numpy as np

from ..clasterization_bold_classifier import ClasterizationBoldClassifier


class MeanBoldClassifier(ClasterizationBoldClassifier):
    def evaluation_method(self, image: np.ndarray) -> float:
        bl_image = self.base_line_image(image)
        image_s = self.get_rid_spaces(bl_image)
        if np.isnan(image_s).all():
            return 0.0
        return image_s.mean()
