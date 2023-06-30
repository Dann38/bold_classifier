from dataset_reader.dataset_reader import Reader
from bold_classifier import PsBoldClassifier, MeanBoldClassifier
import os
from dataset_reader.drawer import Drawer
from clusterizater import BoldSpectralClusterizater

path_data = os.path.join(os.getcwd(), "dataset")
name_data = os.path.join(path_data, "ВКР")

reader = Reader()
pages = reader.get_array_pages(name_data)

ps_classifier = PsBoldClassifier(clusterizater=BoldSpectralClusterizater)
mean_classifier = MeanBoldClassifier(clusterizater=BoldSpectralClusterizater)

# print(ps_classifier.evalusion_on_dataset(pages))

drawer = Drawer()
num_page = 7
rez_ps = ps_classifier.classify(pages[num_page].image, pages[num_page].bboxes)
print(ps_classifier.clusterizater.evalution_listlist(rez_ps, pages[num_page].style))
drawer.imshow(pages[num_page].image, pages[num_page].bboxes, rez_ps)
