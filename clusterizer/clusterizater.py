from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm


class BaseClusterizer(ABC):
    def __init__(self):
        self.significance_level = 0.15

    @abstractmethod
    def clusterize(self, x: np.ndarray) -> np.ndarray:
        pass

    def _is_homogeneous(self, x, x_clust0, x_clust1):
        #  https: // www.tsi.lv / sites / default / files / editor / science / Research_journals / Tr_Tel / 2003 / V1 /
        #  yatskiv_gousarova.pdf
        w1 = np.std(x) * len(x)
        w2 = np.std(x_clust0) * len(x_clust0) + np.std(x_clust1) * len(x_clust1)
        f1 = w2 / w1
        p = 2
        n = len(x)
        za1 = norm.ppf(1 - self.significance_level, loc=0, scale=1)
        f_cr = 1 - 2 / (np.pi * p) - za1 * np.sqrt(abs(2 * (1 - 8 / (np.pi ** 2 / p)) / (n * p)))

        return f1 > f_cr
