import numpy as np

from ..bold_clusterizer import BaseBoldClusterizer


class BoldFixedThresholdClusterizer(BaseBoldClusterizer):
    def _get_clusters(self, x_vectors: np.ndarray) -> np.ndarray:
        # Practice has shown that the border of the bold font is between 0.45 and 0.55
        k_list = np.linspace(0.45, 0.55, 10)
        x = x_vectors[:, 0]
        nearby_x = x_vectors[:, 1]
        std = np.std(x)
        x_cluster_rez = np.zeros_like(x)
        f1_min = 1  # not to be confused with the f1 measure
        for k in k_list:
            x_cluster = np.zeros_like(x)
            x_cluster[nearby_x+std < k] = 1.
            x_cluster[x < k] = 1.
            x_cluster[nearby_x-std > k] = 0.
            f1 = self._get_f1_homogeneous(x, x_cluster)
            if f1 < f1_min:
                x_cluster_rez = x_cluster
                f1_min = f1

        return x_cluster_rez
