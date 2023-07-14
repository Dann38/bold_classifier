import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..bold_clusterizer import BaseBoldClusterizer


class BoldAgglomerativeClusterizer(BaseBoldClusterizer):
    def _get_clusters(self, x_vectors: np.ndarray) -> np.ndarray:
        agg = AgglomerativeClustering()
        agg.fit(x_vectors)
        x_clusters = agg.labels_
        return x_clusters