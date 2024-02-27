import os
from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from bold_classifier import *
from clusterizer import *
from dataset_reader.dataset_reader import Reader
from dataset_reader.page import Page


def get_dataset(name_dataset: str, path_dataset: str) -> List[Page]:
    reader = Reader()
    name_data = os.path.join(path_dataset, name_dataset)
    pages = reader.read_dataset(name_data)
    return pages


def evaluate_on_dataset(classifier: BaseBoldClassifier, pages: List[Page]) -> Dict:
    word_indicators_sum_true = []
    word_indicators_sum = []

    for page in pages:
        list_ = classifier.classify(page.image, page.bboxes)
        list_true = page.style

        word_indicators_true, word_indicators = get_y_true_and_pred(list_true, list_)
        word_indicators_sum.append(word_indicators)
        word_indicators_sum_true.append(word_indicators_true)

    word_indicators_sum_true = np.concatenate(word_indicators_sum_true, axis=None)
    word_indicators_sum = np.concatenate(word_indicators_sum, axis=None)

    evaluate_sum = evaluate_vector(word_indicators_sum_true, word_indicators_sum)
    return evaluate_sum


def evaluate_list(list_true: List[float], list_: List[float]) -> Dict:
    word_indicators_true, word_indicators = get_y_true_and_pred(list_true, list_)
    evaluate = evaluate_vector(word_indicators_true, word_indicators)
    return evaluate


def evaluate_vector(vector_true: np.ndarray, vector_pred: np.ndarray) -> Dict:
    count_word = len(vector_pred)
    p = precision_score(vector_true, vector_pred, zero_division=True)
    r = recall_score(vector_true, vector_pred, zero_division=True)
    f1 = f1_score(vector_true, vector_pred, zero_division=True)
    accuracy = accuracy_score(vector_true, vector_pred)

    return {"count word": count_word, "precession": p, "recall": r, "F1": f1, "accuracy": accuracy}


def get_y_true_and_pred(word_indicators_true: List[float], word_indicators: List[float]) -> (np.ndarray, np.ndarray):
    word_indicators = np.array(word_indicators)
    word_indicators_true = np.array(word_indicators_true)
    #  elements not identified during manual markup
    word_indicators = word_indicators[word_indicators_true != 2]
    word_indicators_true = word_indicators_true[word_indicators_true != 2]
    return word_indicators_true, word_indicators


def print_info(dataset: str, classifier: str):
    print("=" * 60)
    print(f"I N F O : \ndataset:{dataset} \nclassifier: {classifier}")


def print_evaluate(evaluate: Dict) -> None:
    print("\nR E S U L T :")
    for kei in evaluate.keys():
        print(f"{kei + ':':{12}} {evaluate[kei]:.2f}")
    print("-" * 60, "\n")


def check_classifier(classifier: BaseBoldClassifier, pages: List[Page], classifier_name: str,
                     dataset_name: str) -> Dict:
    evaluate_rez = evaluate_on_dataset(classifier, pages)
    print_info(dataset_name, classifier_name)
    print_evaluate(evaluate_rez)
    return evaluate_rez


def check_classifier_and_clusterizer(pages: List[Page], dataset_name: str):
    classifiers = {
        "bold": {"clusterizers": BoldAgglomerativeClusterizer(),
         "classifiers_class": PsBoldClassifier},
         "width": {"clusterizers": BoldAgglomerativeClusterizer(),
         "classifiers_class": PhsBoldClassifier}
    }
    
    best_result = {"classifier": "", "evaluate_rez": {}}
    f1_max = 0.0
    for classifier_name, comb in classifiers.items():
        
        classifier = comb["classifiers_class"](clusterizer=comb["clusterizers"])
        evaluate_rez = check_classifier(classifier, pages, classifier_name, dataset_name)
        if f1_max < evaluate_rez["F1"]:
            f1_max = evaluate_rez["F1"]
            best_result["classifier"] = classifier_name
            best_result["evaluate_rez"] = evaluate_rez
    print("@"*60, "\nBEST RESULT (F1): ")
    print_info(dataset_name, best_result["classifier"])
    print_evaluate(best_result["evaluate_rez"])
    print("@"*60)


def main():
    path_dir_dataset = os.path.join(os.getcwd(), os.path.pardir, "dataset")
    list_dataset_name = ["ВКР", "ГОСТ", "Геометрия", "Разрешение"]  # Prepared sets in the project directory "dataset"
    for dataset_name in list_dataset_name:
        pages = get_dataset(dataset_name, path_dataset=path_dir_dataset)
        check_classifier_and_clusterizer(pages, dataset_name)


if __name__ == "__main__":
    main()
