import numpy as np

from ..clusterizater import BaseClusterizater
from types_font import BOLD, REGULAR


class BoldFixedThresholdClusterizater(BaseClusterizater):
    def clusterization(self, x: np.ndarray) -> np.ndarray:
        k = 0.5
        x_cluster = np.zeros_like(x)
        nearby_x = x.copy()
        nearby_x[:-1] += x[1:]
        nearby_x[1:] += x[:-1]
        nearby_x[0] += x[0]
        nearby_x[-1] += x[-1]
        nearby_x = nearby_x / 3.
        std = np.std(x)
        x_cluster[nearby_x+std < k] = BOLD
        x_cluster[x < k] = BOLD
        x_cluster[nearby_x-std > k] = REGULAR
        return x_cluster
