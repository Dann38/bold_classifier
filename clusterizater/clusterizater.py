from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseClusterizater(ABC):
    @abstractmethod
    def clusterization(self, X: np.ndarray) -> np.ndarray:
        pass

    def listlist2vector(self, listlist: List[List[float]], len_list: List[int]) -> np.ndarray:
        N = sum(len_list)
        vector = np.zeros((N,))
        index = 0
        for line in listlist:
            for i in range(len(line)):
                vector[index] = line[i]
                index += 1
        return vector

    def vector2listlist(self, vector: np.ndarray, len_list: List[int]) -> List[List[float]]:
        listlist = []
        index = 0
        for i in range(len(len_list)):
            listlist.append([])
            for j in range(len_list[i]):
                listlist[i].append(vector[index])
                index += 1
        return listlist

    def evalution_listlist(self, listlist: List[List[float]], listlist_true: List[List[float]]) -> Dict:
        len_lines = [len(line) for line in listlist_true]
        X = self.listlist2vector(listlist, len_lines)
        X_true = self.listlist2vector(listlist_true, len_lines)

        N = len(X)

        TX = X[X == X_true]
        FX = X[X != X_true]
        TP = sum(TX)
        TN = len(TX) - TP
        FP = sum(FX)
        FN = len(FX) - FP

        if TP+FP == 0:
            p = 1.
        else:
            p = TP / (TP + FP)
        if TP + FN == 0:
            r = 1.
        else:
            r = TP / (TP + FN)

        if p + r == 0:
            f1 = 0.
        else:
            f1 = 2 * (p * r) / (p + r)
        accuracy = (TP + TN) / N

        return {"N": N, "precession": p, "recall": r, "F1": f1, "accuracy": accuracy}
