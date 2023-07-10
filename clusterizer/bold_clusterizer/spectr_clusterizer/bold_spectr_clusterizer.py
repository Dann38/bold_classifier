import numpy as np
from sklearn.cluster import SpectralClustering

from ..bold_clusterizer import BaseBoldClusterizer


class BoldSpectralClusterizer(BaseBoldClusterizer):
    def _get_clusters(self, x_vectors: np.ndarray) -> np.ndarray:
        spectr = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0)
        spectr.fit(x_vectors)
        x_clusters = spectr.labels_
        return x_clusters

