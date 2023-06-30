from abc import abstractmethod
from ..bold_classifier import BaseBoldClassifier, BOLD, REGULAR
from clusterizater import Bold2MeanClusterizater
from typing import List, Dict
import numpy as np
from dataset_reader.bbox import BBox  # TODO Изменить путь
from dataset_reader.page import Page


class ClasterizationBoldClassifier(BaseBoldClassifier):
    def __init__(self, clusterizater=Bold2MeanClusterizater):
        self.clusterizater = clusterizater()

    def classify(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        processed_image = self.preprocessing(image)
        lines_estimates = self.get_evaluation_bboxes(processed_image, bboxes)
        lines_bold_indicators = self.clasterization(lines_estimates)
        return lines_bold_indicators

    @abstractmethod
    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        pass

    def get_evaluation_bboxes(self, image: np.ndarray,
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

    def clasterization(self, lines_estimates: List[List[float]]) -> List[List[float]]:
        len_lines = [len(line) for line in lines_estimates]
        X = self.clusterizater.listlist2vector(lines_estimates, len_lines)
        X_cluster = self.clusterizater.clusterization(X)
        lines_estimates = self.clusterizater.vector2listlist(X_cluster, len_lines)
        return lines_estimates

    # Возращение результатов без кластеризации (полезна при отладке)
    def get_lines_estimates(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        processed_image = self.preprocessing(image)
        lines_estimates = self.get_evaluation_bboxes(processed_image, bboxes)
        return lines_estimates

    def evalusion_on_dataset(self, pages: List[Page]) -> Dict:
        count_pages = len(pages)
        evalution_sum = {
            "precession": 0,
            "recall": 0,
            "F1": 0,
            "accuracy": 0,
            "N": 0
        }

        for num_page in range(count_pages):
            listlist = self.classify(pages[num_page].image, pages[num_page].bboxes)
            evalution_method = self.clusterizater.evalution_listlist(listlist, pages[num_page].style)
            print(evalution_method)
            for key in evalution_sum.keys():
                if key == "N":
                    evalution_sum[key] += evalution_method["N"]
                else:
                    evalution_sum[key] += evalution_method["N"] * evalution_method[key]

        for key in evalution_sum.keys():
            if key != "N":
                evalution_sum[key] = evalution_sum[key] / evalution_sum["N"]

        return evalution_sum
