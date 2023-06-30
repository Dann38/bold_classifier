from dataset_reader.dataset_reader import Reader
from bold_classifier import PsBoldClassifier, MeanBoldClassifier
import os


path_data = os.path.join(os.getcwd(), "dataset")
name_data = os.path.join(path_data, "ГОСТ")

reader = Reader()
pages = reader.get_array_pages(name_data)

ps_classifier = PsBoldClassifier()
mean_classifier = MeanBoldClassifier()

print(mean_classifier.evalusion_on_dataset(pages))

# drawer = Drawer()
# num_page = 2
# rez_ps = ps_classifier.classify(pages[num_page].image, pages[num_page].bboxes)
# print(ps_classifier.clusterizater.evalution_listlist(rez_ps, pages[num_page].style))
# drawer.imshow(pages[num_page].image, pages[num_page].bboxes, rez_ps)
