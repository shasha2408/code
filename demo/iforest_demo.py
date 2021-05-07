import sys
import logging
from pyod.models.iforest import IForest

sys.path.append("../")

from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint
from common.evaluation import evaluate_all

dataset = "SMD"
subdataset = "machine-1-1"
n_estimators = 100
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    # load dataset
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    od = IForest(n_estimators=n_estimators)

    od.fit(x_train)

    anomaly_score = od.decision_function(x_test)

    anomaly_label = x_test_labels

    # Make evaluation
    evaluate_all(anomaly_score, anomaly_label)