import numpy as np
from typing import List
PERMISSIBLE_H_BBOX = 5


def get_rid_spaces(image: np.ndarray) -> np.ndarray:
    x = image.mean(0)
    return image[:, x < 0.95]


def base_line_image(image: np.ndarray) -> np.ndarray:
    h = image.shape[0]
    if h < PERMISSIBLE_H_BBOX:
        return image
    mean_ = image.mean(1)
    dmean = abs(mean_[:-1] - mean_[1:])

    max1 = 0
    max2 = 0
    argmax1 = 0
    argmax2 = 0
    for i in range(len(dmean)):
        if dmean[i] > max2:
            if dmean[i] > max1:
                max2 = max1
                argmax2 = argmax1
                max1 = dmean[i]
                argmax1 = i
            else:
                max2 = dmean[i]
                argmax2 = i
    h_min = min(argmax1, argmax2)
    h_max = max(argmax1, argmax2)

    return image[h_min:h_max + 1, :]


def listlist2vector(listlist: List[List[float]], len_list: List[int]) -> np.ndarray:
    N = sum(len_list)
    vector = np.zeros((N,))
    index = 0
    for line in listlist:
        for i in range(len(line)):
            vector[index] = line[i]
            index += 1
    return vector


def vector2listlist(vector: np.ndarray, len_list: List[int]) -> List[List[float]]:
    listlist = []
    index = 0
    for i in range(len(len_list)):
        listlist.append([])
        for j in range(len_list[i]):
            listlist[i].append(vector[index])
            index += 1
    return listlist