from typing import List
import numpy as np
import os
import cv2
import pickle

from .bbox import BBox
from .page import Page


class Reader:
    def get_array_pages(self, path_dir: str) -> List[Page]:
        files = os.listdir(path_dir)
        pages = [self.get_page(path_dir, name_file) for name_file in files if self.__is_page(path_dir, name_file)]
        return pages

    def __is_page(self, path_dir: str, name: str) -> bool:
        if not name.split(".")[-1] in ["jpg", "png", "jpeg"]:
            return False

        img_path = os.path.join(path_dir, name)
        if not os.path.isfile(img_path):
            return False

        pkl_path = os.path.join(path_dir, name + ".pkl")
        if not os.path.isfile(pkl_path):
            return False

        return True

    def get_page(self, path_dir: str, name: str) -> Page:
        img_path = os.path.join(path_dir, name)
        pkl_path = os.path.join(path_dir, name + ".pkl")

        image = self._get_image(path_dir, img_path)
        bboxes, style = self.__get_bboxes_and_style(path_dir, pkl_path)
        page = Page(image, bboxes, style, name)
        return page

    def _get_image(self, path_dir: str, name_image_file: str) -> np.ndarray:
        path_image = os.path.join(path_dir, name_image_file)
        with open(path_image, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        image = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        return image

    def __get_bboxes_and_style(self, path_dir: str,
                              name_pkl_file: str) -> (List[List[BBox]], List[List[float]]):
        path_image = os.path.join(path_dir, name_pkl_file)
        with open(path_image, 'rb') as f:
            (dict_lines, style) = pickle.load(f)
        bboxes = [[BBox.from_dict(box) for box in line] for line in dict_lines]
        return bboxes, style
