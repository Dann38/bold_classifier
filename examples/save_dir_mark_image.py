import os

from bold_classifier import PsBoldClassifier
from dataset_reader.dataset_reader import Reader
from dataset_reader.drawer import Drawer

path_dir_rez = "rez_ВКР"
name_dataset = "ВКР"
classifier_class = PsBoldClassifier
os.mkdir(path_dir_rez)

path_data = os.path.join(os.getcwd(), os.path.pardir, "dataset")
name_data = os.path.join(path_data, name_dataset)

reader = Reader()

pages = reader.read_dataset(name_data)
drawer = Drawer()
for page in pages:
    classifier = classifier_class()
    rez = classifier.classify(page.image, page.bboxes)
    drawer.imsave(page.image, page.bboxes, rez, os.path.join(path_dir_rez, "rez_"+page.name))
