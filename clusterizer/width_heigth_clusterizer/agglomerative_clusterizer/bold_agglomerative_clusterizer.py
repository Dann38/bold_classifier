import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..width_heigth_clusterizer import BaseWidthHeigthClusterizer


class WidthHeigthAgglomerativeClusterizer(BaseWidthHeigthClusterizer):
    def _get_clusters(self, x_vectors: np.ndarray) -> np.ndarray:
        agg = AgglomerativeClustering()
        agg.fit(x_vectors)
        x_clusters = agg.labels_
        return x_clusters