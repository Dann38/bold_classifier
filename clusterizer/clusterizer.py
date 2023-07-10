from abc import ABC, abstractmethod

import numpy as np


class BaseClusterizer(ABC):
    @abstractmethod
    def clusterize(self, x: np.ndarray) -> np.ndarray:
        pass

