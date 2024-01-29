import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..bold_clusterizer import BaseBoldClusterizer


class NAgglomerativeClusterizer(BaseBoldClusterizer):
    def _get_clusters(self, x_vectors: np.ndarray) -> np.ndarray:
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2)
        agg2 = AgglomerativeClustering(n_clusters=2)

        agg.fit(x_vectors)
        x_clusters = agg.labels_

        clusts = np.unique(x_clusters)
        ps_vec = x_vectors[:, 0]
        ps_clusts = []
        for i, c in enumerate(clusts):
            ps_clusts.append(ps_vec[x_clusters==c].mean())
            
        ps_clusts = np.array(ps_clusts)
        clusters2 = agg2.fit(ps_clusts.reshape(-1, 1) )

        for b in clusts[clusters2.labels_ == 1]:
            x_clusters[x_clusters == b] = 1
        for r in clusts[clusters2.labels_ == 0]:
            x_clusters[x_clusters == r] = 0

        return x_clusters