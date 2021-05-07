import sys
import logging
from pyod.models.knn import KNN

sys.path.append("../")

from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint
from common.evaluation import evaluate_all

dataset = "SMD"
subdataset = "machine-1-1"
n_neighbors = 5
radius = 1.0
leaf_size = 30
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
    )

    # load dataset
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    # data preprocessing for MSCRED
    od = KNN(n_neighbors=n_neighbors, radius=radius, leaf_size=leaf_size)
    od.fit(x_train)

    # get outlier scores
    anomaly_score = od.decision_function(x_test)

    anomaly_label = x_test_labels

    # Make evaluation
    evaluate_all(anomaly_score, anomaly_label)
