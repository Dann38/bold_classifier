from abc import abstractmethod
from typing import List

import numpy as np

from binarizer import ValleyEmphasisBinarizer
from dataset_reader.bbox import BBox
from ..bold_classifier import BaseBoldClassifier
from ..types_font import REGULAR, BOLD

PERMISSIBLE_H_BBOX = 5  # that height bbox after which it makes no sense сrop bbox
PERMISSIBLE_W_BBOX = 3

class CNNBoldClassifier(BaseBoldClassifier):
    def __init__(self):
        self.binarizer = ValleyEmphasisBinarizer()

    def classify(self, image: np.ndarray,  bboxes: List[BBox]) -> List[float]:
        gray_image = self.binarizer.binarize(image)
        if len(bboxes) == 0:
            return []
        if len(bboxes) == 1:
            return [REGULAR]
        bboxes_evaluation = self.get_bboxes_evaluation(gray_image, bboxes)
        bboxes_indicators = [REGULAR if e < 0.5 else BOLD for e in bboxes_evaluation]
        print(bboxes_evaluation)
        return bboxes_indicators
    
    def get_bboxes_evaluation(self, image: np.ndarray, bboxes: List[BBox]) -> List[float]:
        list_evaluation = []
        for bbox in bboxes:
            word_image = image[bbox.y_top_left:bbox.y_bottom_right, bbox.x_top_left:bbox.x_bottom_right]
            evaluation = self.get_word_evaluation(word_image)
            list_evaluation.append(evaluation)
        return list_evaluation

    @abstractmethod
    def get_word_evaluation(self, word_image: np.ndarray) -> float:
        pass