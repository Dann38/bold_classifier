import numpy as np
from typing import List
from sklearn.cluster import KMeans

from ..bold_clusterizer import BaseBoldClusterizer


class Bold2MeanClusterizer(BaseBoldClusterizer):
    def _get_clusters(self, x_vectors: np.ndarray) -> np.ndarray:
        kmeans = KMeans(n_clusters=2, n_init="auto")
        kmeans.fit(x_vectors)
        x_clusters = kmeans.labels_
        return x_clusters
