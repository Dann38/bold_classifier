import numpy as np
from sklearn.cluster import SpectralClustering

from ..clusterizater import BaseClusterizer
from types_font import BOLD, REGULAR


class BoldSpectralClusterizer(BaseClusterizer):
    def clusterize(self, x: np.ndarray) -> np.ndarray:
        nearby_x = x.copy()
        nearby_x[:-1] += x[1:]
        nearby_x[1:] += x[:-1]
        nearby_x[0] += x[0]
        nearby_x[-1] += x[-1]
        nearby_x = nearby_x / 3.

        x_vec = [[x[i], nearby_x[i]] for i in range(len(x))]
        spectr = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0)
        spectr.fit(x_vec)
        x_clust = spectr.labels_

        x_clust0 = x[x_clust == 0]
        x_clust1 = x[x_clust == 1]

        if self._is_homogeneous(x, x_clust0, x_clust1):
            return np.zeros_like(x)+REGULAR
        if np.mean(x[x_clust == 1]) < np.mean(x[x_clust == 0]):
            x_clust[x_clust == 1] = BOLD
            x_clust[x_clust == 0] = REGULAR
        else:
            x_clust[x_clust == 0] = BOLD
            x_clust[x_clust == 1] = REGULAR

        return x_clust
