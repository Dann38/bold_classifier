from abc import abstractmethod
from ..bold_classifier import BaseBoldClassifier, BOLD, REGULAR
from .utils import listlist2vector, vector2listlist
from typing import List
import numpy as np
from dataset_reader.bbox import BBox  # TODO Изменить путь
from sklearn.cluster import KMeans


class ClasterizationBoldClassifier(BaseBoldClassifier):
    def __init__(self):
        self.k0 = 0.5
        self.std = 0.1

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
        X = listlist2vector(lines_estimates, len_lines)
        X = [[x] for x in X]
        kmeans = KMeans(n_clusters=2, n_init="auto")
        kmeans.fit(X)
        X = kmeans.labels_

        cluster0 = kmeans.cluster_centers_[0][0]
        cluster1 = kmeans.cluster_centers_[1][0]
        print(f"cluster 0:{cluster0}, clusrer 1:{cluster1}")
        center_cluster = (cluster0+cluster1)/2
        bold_cluster = min(cluster0, cluster1)
        regular_cluster = max(cluster0, cluster1)
        if cluster0 == bold_cluster:
            X = 1.0 - X
        if bold_cluster > self.k0+self.std:
            X = np.zeros_like(X)
        elif regular_cluster < self.k0-self.std:
            X = np.ones_like(X)
        lines_bold_indicators = vector2listlist(X, len_lines)
        return lines_bold_indicators


    # Возращение результатов без кластеризации (полезна при отладке)
    def get_lines_estimates(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        processed_image = self.preprocessing(image)
        lines_estimates = self.get_evaluation_bboxes(processed_image, bboxes)
        return lines_estimates






