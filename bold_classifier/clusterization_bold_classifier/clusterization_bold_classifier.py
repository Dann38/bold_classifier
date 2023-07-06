from abc import abstractmethod
from typing import List
import numpy as np

from ..bold_classifier import BaseBoldClassifier
from ..utils import vector2llist, llist2vector

from clusterizer import BoldSpectralClusterizer, BaseClusterizer
from binarizer import ValleyEmphasisBinarizer
from dataset_reader.bbox import BBox

PERMISSIBLE_H_BBOX = 5  # that height bbox after which it makes no sense сrop bbox


class ClusterizationBoldClassifier(BaseBoldClassifier):
    def __init__(self, clusterizer: BaseClusterizer = BoldSpectralClusterizer()):
        self.binarizer = ValleyEmphasisBinarizer()
        self.clusterizer = clusterizer

    def classify(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        lines_estimates = self.get_lines_estimates(image, bboxes)
        lines_bold_indicators = self.__clusterize(lines_estimates)
        return lines_bold_indicators

    def get_lines_estimates(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        processed_image = self._preprocessing(image)
        lines_estimates = self.__get_evaluation_bboxes(processed_image, bboxes)
        return lines_estimates

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        return self.binarizer.binarize(image)

    def __get_evaluation_bboxes(self, image: np.ndarray,
                               bboxes: List[List[BBox]]) -> List[List[float]]:
        evaluation_bboxes = []
        for line in bboxes:
            evaluation_bboxes.append([])
            for bbox in line:
                image_bbox = image[bbox.y_top_left:bbox.y_bottom_right,
                                   bbox.x_top_left:bbox.x_bottom_right]
                evaluation_bbox = self.evaluation_method(image_bbox)
                evaluation_bboxes[-1].append(evaluation_bbox)
        return evaluation_bboxes

    @abstractmethod
    def evaluation_method(self, image: np.ndarray) -> float:
        pass

    def __clusterize(self, lines_estimates: List[List[float]]) -> List[List[float]]:
        len_lines = [len(line) for line in lines_estimates]
        word_estimates = llist2vector(lines_estimates, len_lines)
        word_indicators = self.clusterizer.clusterize(word_estimates)
        lines_estimates = vector2llist(word_indicators, len_lines)
        return lines_estimates

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

        return image[h_min:h_max, :]

