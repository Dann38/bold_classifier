import os

from bold_classifier import PsWidthHeigthClassifier
from dataset_reader.dataset_reader import Reader
from dataset_reader.drawer import Drawer

path_data = os.path.join(os.getcwd(),  os.path.pardir, "dataset")
name_data = os.path.join(path_data, "ВКР")
num_page = 1

reader = Reader()
pages = reader.read_dataset(name_data)
drawer = Drawer()
classifier = PsWidthHeigthClassifier()

rez = classifier.classify(pages[num_page].image, pages[num_page].bboxes)
drawer.imshow(pages[num_page].image, pages[num_page].bboxes, rez)
