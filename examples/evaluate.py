import os
from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset_reader.dataset_reader import Reader
from dataset_reader.page import Page

from bold_classifier import *
from bold_classifier.utils import llist2vector
from bold_classifier.bold_classifier import BaseBoldClassifier


path_dir_dataset = os.path.join(os.getcwd(), "dataset")
list_name_dataset = ["ВКР", "ГОСТ", "Геометрия"]
classifiers = {
    "PsBoldClassifier": PsBoldClassifier(),
    "MeanBoldClassifier": MeanBoldClassifier(),
    "HistBoldClassifier": HistBoldClassifier()
}


def get_dataset(name_dataset: str, path_dataset: str = path_dir_dataset) -> List[Page]:
    reader = Reader()
    name_data = os.path.join(path_dataset, name_dataset)
    pages = reader.get_array_pages(name_data)
    return pages


def evaluate_on_dataset(classifier: BaseBoldClassifier, pages: List[Page]) -> Dict:
    word_indicators_sum_true = []
    word_indicators_sum = []

    for page in pages:
        llist = classifier.classify(page.image, page.bboxes)
        llist_true = page.style

        word_indicators_true, word_indicators = get_y_true_and_pred(llist_true, llist)
        word_indicators_sum.append(word_indicators)
        word_indicators_sum_true.append(word_indicators_true)

    word_indicators_sum_true = np.concatenate(word_indicators_sum_true, axis=None)
    word_indicators_sum = np.concatenate(word_indicators_sum, axis=None)

    evaluate_sum = evaluate_vector(word_indicators_sum_true, word_indicators_sum)
    return evaluate_sum


def evaluate_llist(llist_true: List[List[float]], llist: List[List[float]]) -> Dict:
    word_indicators_true, word_indicators = get_y_true_and_pred(llist_true, llist)
    evaluate = evaluate_vector(word_indicators_true, word_indicators)
    return evaluate


def evaluate_vector(vector_true: np.ndarray, vector_pred: np.ndarray) -> Dict:
    count_word = len(vector_pred)
    p = precision_score(vector_true, vector_pred, zero_division=True)
    r = recall_score(vector_true, vector_pred, zero_division=True)
    f1 = f1_score(vector_true, vector_pred, zero_division=True)
    accuracy = accuracy_score(vector_true, vector_pred)

    return {"count word": count_word, "precession": p, "recall": r, "F1": f1, "accuracy": accuracy}


def get_y_true_and_pred(llist_true: List[List[float]], llist: List[List[float]]) -> (List[float], List[float]):
    len_lines = [len(line) for line in llist_true]
    word_indicators = llist2vector(llist, len_lines)
    word_indicators_true = llist2vector(llist_true, len_lines)

    #  elements not identified during manual markup
    word_indicators = word_indicators[word_indicators_true != 2]
    word_indicators_true = word_indicators_true[word_indicators_true != 2]
    return word_indicators_true, word_indicators


def print_evaluate(evaluate: Dict) -> None:
    for kei in evaluate.keys():
        print(f"{kei + ':':{12}} {evaluate[kei]:.2f}")


def main():
    for name_dataset in list_name_dataset:
        pages = get_dataset(name_dataset)
        print("*" * 15, name_dataset, "*" * 15)
        for classifier_name, classifier in classifiers.items():
            evaluate_rez = evaluate_on_dataset(classifier, pages)
            print("-" * 10, classifier_name, "-" * 10)
            print_evaluate(evaluate_rez)


if __name__ == "__main__":
    main()