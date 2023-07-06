from abc import abstractmethod
from typing import List
import numpy as np

from ..bold_classifier import BaseBoldClassifier
from ..utils import vector2llist, llist2vector

from clusterizer import Bold2MeanClusterizer, BaseClusterizer
from binarizer import ValleyEmphasisBinarizer
from dataset_reader.bbox import BBox

PERMISSIBLE_H_BBOX = 5  # that height bbox after which it makes no sense сrop bbox


class ClusterizationBoldClassifier(BaseBoldClassifier):
    def __init__(self, clusterizer: BaseClusterizer = Bold2MeanClusterizer()):
        self.binarizer = ValleyEmphasisBinarizer()
        self.clusterizer = clusterizer

    def classify(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        bboxes_evaluation = self.get_bboxes_evaluation(image, bboxes)
        bboxes_indicators = self.__clusterize(bboxes_evaluation)
        return bboxes_indicators

    def get_bboxes_evaluation(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        processed_image = self._preprocessing(image)
        bboxes_evaluation = self.__get_evaluation_bboxes(processed_image, bboxes)
        return bboxes_evaluation

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        return self.binarizer.binarize(image)

    def __get_evaluation_bboxes(self, image: np.ndarray, bboxes: List[List[BBox]]) -> List[List[float]]:
        bboxes_evaluation = []
        for line in bboxes:
            bboxes_evaluation.append([])
            for bbox in line:
                bbox_image = image[bbox.y_top_left:bbox.y_bottom_right,
                                   bbox.x_top_left:bbox.x_bottom_right]
                bbox_evaluation = self.evaluation_one_bbox_image(bbox_image)
                bboxes_evaluation[-1].append(bbox_evaluation)
        return bboxes_evaluation

    @abstractmethod
    def evaluation_one_bbox_image(self, image: np.ndarray) -> float:
        pass

    def __clusterize(self, bboxes_evaluation: List[List[float]]) -> List[List[float]]:
        len_lines = [len(line) for line in bboxes_evaluation]
        vector_bbox_evaluation = llist2vector(bboxes_evaluation, len_lines)
        vector_bbox_indicators = self.clusterizer.clusterize(vector_bbox_evaluation)
        bboxes_indicators = vector2llist(vector_bbox_indicators, len_lines)
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
        for i in range(len(delta_mean)):
            if delta_mean[i] > max2:
                if delta_mean[i] > max1:
                    max2 = max1
                    argmax2 = argmax1
                    max1 = delta_mean[i]
                    argmax1 = i
                else:
                    max2 = delta_mean[i]
                    argmax2 = i
        h_min = min(argmax1, argmax2)
        h_max = min(max(argmax1, argmax2) + 1, h)
        if h_max-h_min < PERMISSIBLE_H_BBOX:
            return image
        return image[h_min:h_max, :]

