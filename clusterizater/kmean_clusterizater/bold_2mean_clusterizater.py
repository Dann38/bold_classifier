import numpy as np
from sklearn.cluster import KMeans

from ..clusterizater import BaseClusterizater
from bold_classifier import BOLD, REGULAR


class Bold2MeanClusterizater(BaseClusterizater):
    def __init__(self, significance_level=0.1):
        self.significance_level = significance_level

    def clusterization(self, x: np.ndarray) -> np.ndarray:
        nearby_x = x.copy()
        nearby_x[:-1] += x[1:]
        nearby_x[1:] += x[:-1]
        nearby_x[0] += x[0]
        nearby_x[-1] += x[-1]
        nearby_x = nearby_x / 3.
        x_vec = [[x[i], nearby_x[i]] for i in range(len(x))]
        kmeans = KMeans(n_clusters=2, n_init="auto")
        kmeans.fit(x_vec)

        cluster0 = kmeans.cluster_centers_[0][0]
        cluster1 = kmeans.cluster_centers_[1][0]

        bold_cluster = min(cluster0, cluster1)
        regular_cluster = max(cluster0, cluster1)
        distance_cluster = regular_cluster-bold_cluster
        x_clust = np.zeros_like(x) + REGULAR
        x_clust[x-bold_cluster < bold_cluster+distance_cluster*self.significance_level] = BOLD
        return x_clust
