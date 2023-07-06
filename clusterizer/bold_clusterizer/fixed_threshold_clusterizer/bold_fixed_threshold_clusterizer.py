import numpy as np
from ..bold_clusterizer import BaseBoldClusterizer


class BoldFixedThresholdClusterizer(BaseBoldClusterizer):
    def _get_clusters(self, x_vectors: np.ndarray) -> np.ndarray:
        k = 0.5
        x = x_vectors[:, 0]
        nearby_x = x_vectors[:, 1]
        std = np.std(x)

        x_cluster = np.zeros_like(x)
        x_cluster[nearby_x+std < k] = 1
        x_cluster[x < k] = 1
        x_cluster[nearby_x-std > k] = 0
        return x_cluster
