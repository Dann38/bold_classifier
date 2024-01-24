from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from binarizer import ValleyEmphasisBinarizer
from clusterizer import WidthHeigthAgglomerativeClusterizer, BaseClusterizer
from dataset_reader.bbox import BBox
from ..bold_classifier import BaseBoldClassifier
from ..types_font import REGULAR



class ClusterizationWidthHeigthClassifier(BaseBoldClassifier):
    def __init__(self, clusterizer: BaseClusterizer = None):
        self.binarizer = ValleyEmphasisBinarizer()
        if clusterizer is None:
            self.clusterizer = WidthHeigthAgglomerativeClusterizer()
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
    
    def get_bboxes_evaluation(self, image: np.ndarray,  bboxes: List[BBox]) -> List[Tuple[float, float]]:
        processed_image = self._preprocessing(image)
        bboxes_evaluation = self.__get_evaluation_bboxes(processed_image, bboxes)
        return bboxes_evaluation

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        return self.binarizer.binarize(image)

    def __get_evaluation_bboxes(self, image: np.ndarray, bboxes: List[BBox]) -> List[Tuple[float, float]]:
        bboxes_evaluation = [self.__evaluation_one_bbox(image, bbox) for bbox in bboxes]
        return bboxes_evaluation

    @abstractmethod
    def evaluation_one_bbox_image(self, image: np.ndarray) -> Tuple[float, float]:
        pass

    def __evaluation_one_bbox(self, image: np.ndarray, bbox: BBox) -> Tuple[float, float]:
        bbox_image = image[bbox.y_top_left:bbox.y_bottom_right, bbox.x_top_left:bbox.x_bottom_right]
        return self.evaluation_one_bbox_image(bbox_image) if self._is_correct_bbox_image(bbox_image) else (0.0, 0.0)

    def __clusterize(self, bboxes_evaluation: List[Tuple[float, float]]) -> List[float]:
        
        vector_bbox_evaluation = np.array( bboxes_evaluation)
        w1, h1 = vector_bbox_evaluation.max(axis=0)
        vector_bbox_evaluation[:, 0] = vector_bbox_evaluation[:, 0]/w1
        vector_bbox_evaluation[:, 1] = vector_bbox_evaluation[:, 1]/h1
        vector_bbox_indicators = self.clusterizer.clusterize(vector_bbox_evaluation)
        bboxes_indicators = list(vector_bbox_indicators)
        return bboxes_indicators