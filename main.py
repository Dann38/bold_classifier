from dataset_reader.dataset_reader import Reader
from bold_classifier import PsBoldClassifier, MeanBoldClassifier
from bold_classifier.clasterization_bold_classifier.utils import listlist2vector
import os
from dataset_reader.drawer import Drawer

def evalution(X, X_true):
    N = len(X)

    TX = X[X == X_true]
    FX = X[X != X_true]
    TP = sum(TX)
    TN = len(TX) - TP
    FP = sum(FX)
    FN = len(FX) - FP

    p = TP/(TP+FP)
    if TP+FN == 0:
        r = 1.
    else:
        r = TP/(TP+FN)

    if p+r == 0:
        f1 = 0.
    else:
        f1 = 2*(p*r)/(p+r)
    accuracy = (TP+TN)/N

    return {"precession": p, "recall": r, "F1": f1, "accuracy": accuracy}



path_data = os.path.join(os.getcwd(), "dataset")
geom_data = os.path.join(path_data, "ГОСТ")

reader = Reader()
pages = reader.get_array_pages(geom_data)
count_pages = len(pages)

ps_classifier = PsBoldClassifier()
mean_classifier = MeanBoldClassifier()

c_sum = 0
evalution_sum = {
    "precession": 0,
    "recall": 0,
    "F1": 0,
    "accuracy": 0
}

for num_page in range(count_pages):
    rez_ps = ps_classifier.classify(pages[num_page].image, pages[num_page].bboxes)
    rez_true = pages[num_page].style

    len_lines = [len(line) for line in rez_true]
    X = listlist2vector(rez_ps, len_lines)
    X_true = listlist2vector(rez_true, len_lines)

    c = len(X_true)
    c_sum += c

    evalution_method = evalution(X, X_true)
    for key in evalution_sum.keys():
        evalution_sum[key] += c*evalution_method[key]

for key in evalution_sum.keys():
    evalution_sum[key] = evalution_sum[key]/c_sum

print(evalution_sum)

# drawer = Drawer()
# num_page = 6
# rez_ps = ps_classifier.classify(pages[num_page].image, pages[num_page].bboxes)
# drawer.imshow(pages[num_page].image, pages[num_page].bboxes, rez_ps)
