from abc import ABC, abstractmethod
from typing import List

import numpy as np

from dataset_reader.bbox import BBox


class BaseBoldClassifier(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        pass


