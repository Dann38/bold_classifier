from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseClusterizater(ABC):
    @abstractmethod
    def clusterization(self, X: np.ndarray) -> np.ndarray:
        pass
