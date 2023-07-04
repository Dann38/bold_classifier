from abc import ABC, abstractmethod
import numpy as np


class BaseClusterizater(ABC):
    @abstractmethod
    def clusterization(self, x: np.ndarray) -> np.ndarray:
        pass
