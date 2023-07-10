from typing import List

import numpy as np

from .bbox import BBox


class Page:
    def __init__(self, image: np.ndarray, bboxes: List[List[BBox]],
                 style: List[List[int]], name: str):
        self.image = image
        self.bboxes = bboxes
        self.style = style
        self.name = name
