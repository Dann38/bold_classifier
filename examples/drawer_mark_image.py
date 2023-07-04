from dataset_reader.dataset_reader import Reader
import os
from dataset_reader.drawer import Drawer
from bold_classifier import PsBoldClassifier


path_data = os.path.join(os.getcwd(), "../dataset")
name_data = os.path.join(path_data, "ВКР")
num_page = 7

reader = Reader()
pages = reader.get_array_pages(name_data)
drawer = Drawer()
classifier = PsBoldClassifier()

rez = classifier.classify(pages[num_page].image, pages[num_page].bboxes)
drawer.imshow(pages[num_page].image, pages[num_page].bboxes, rez)
