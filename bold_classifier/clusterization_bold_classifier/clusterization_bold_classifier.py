from abc import abstractmethod
from typing import List

import numpy as np

from binarizer import ValleyEmphasisBinarizer
from clusterizer import Bold2MeanClusterizer, BaseClusterizer
from dataset_reader.bbox import BBox
from ..bold_classifier import BaseBoldClassifier
from ..types_font import REGULAR

PERMISSIBLE_H_BBOX = 5  # that height bbox after which it makes no sense сrop bbox


class ClusterizationBoldClassifier(BaseBoldClassifier):
    def __init__(self, clusterizer: BaseClusterizer = None):
        self.binarizer = ValleyEmphasisBinarizer()
        if clusterizer is None:
            self.clusterizer = Bold2MeanClusterizer()
        else:
            self.clusterizer = clusterizer

    def classify(self, image: np.ndarray,  bboxes: List[BBox]) -> List[float]:
        if len(bboxes) == 0:
            return []
        if len(bboxes) == 1:
            return [REGULAR]
        bboxes_evaluation = self.get_bboxes_evaluation(image, bboxes)
        bboxes_indicators = self.__clusterize(bboxes_evaluation)
        return bboxes_indicators

    def get_bboxes_evaluation(self, image: np.ndarray,  bboxes: List[BBox]) -> List[float]:
        processed_image = self._preprocessing(image)
        bboxes_evaluation = self.__get_evaluation_bboxes(processed_image, bboxes)
        return bboxes_evaluation

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        return self.binarizer.binarize(image)

    def __get_evaluation_bboxes(self, image: np.ndarray, bboxes: List[BBox]) -> List[float]:
        bboxes_evaluation = [self.__evaluation_one_bbox(image, bbox) for bbox in bboxes]
        return bboxes_evaluation

    @abstractmethod
    def evaluation_one_bbox_image(self, image: np.ndarray) -> float:
        pass

    def __evaluation_one_bbox(self, image: np.ndarray, bbox: BBox) -> float:
        bbox_image = image[bbox.y_top_left:bbox.y_bottom_right, bbox.x_top_left:bbox.x_bottom_right]
        if self._is_correct_bbox_image(bbox_image):
            return self.evaluation_one_bbox_image(bbox_image)
        else:
            return 1.

    def __clusterize(self, bboxes_evaluation: List[float]) -> List[float]:
        vector_bbox_evaluation = np.array(bboxes_evaluation)
        vector_bbox_indicators = self.clusterizer.clusterize(vector_bbox_evaluation)
        bboxes_indicators = list(vector_bbox_indicators)
        return bboxes_indicators

    def _get_rid_spaces(self, image: np.ndarray) -> np.ndarray:
        x = image.mean(0)
        return image[:, x < 0.95]

    def _get_base_line_image(self, image: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        if h < PERMISSIBLE_H_BBOX:
            return image
        mean_ = image.mean(1)
        delta_mean = abs(mean_[:-1] - mean_[1:])

        max1 = 0
        max2 = 0
        argmax1 = 0
        argmax2 = 0
        for i, delta_mean_i in enumerate(delta_mean):
            if delta_mean_i <= max2:
                continue
            if delta_mean_i > max1:
                max2 = max1
                argmax2 = argmax1
                max1 = delta_mean_i
                argmax1 = i
            else:
                max2 = delta_mean_i
                argmax2 = i
        h_min = min(argmax1, argmax2)
        h_max = min(max(argmax1, argmax2) + 1, h)
        if h_max-h_min < PERMISSIBLE_H_BBOX:
            return image
        return image[h_min:h_max, :]

    def _is_correct_bbox_image(self, image: np.ndarray) -> bool:
        h, w = image.shape[0:2]
        return h > 3 and w > 3
