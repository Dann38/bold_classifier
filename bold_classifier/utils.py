from typing import List

import numpy as np


def llist2vector(llist: List[List[float]], len_list: List[int]) -> np.ndarray:
    len_vector = sum(len_list)
    vector = np.zeros((len_vector,))

    index = 0
    for line in llist:
        for i in range(len(line)):
            vector[index] = line[i]
            index += 1
    return vector


def vector2llist(vector: np.ndarray, len_list: List[int]) -> List[List[float]]:
    llist = []
    index = 0
    for i in range(len(len_list)):
        llist.append([])
        for j in range(len_list[i]):
            llist[i].append(vector[index])
            index += 1
    return llist
