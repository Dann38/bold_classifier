from abc import ABC, abstractmethod
from typing import List
from dataset_reader.bbox import BBox  # TODO Изменить путь
import numpy as np

BOLD = 1.0
REGULAR = 0.0


class BaseBoldClassifier(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        pass


