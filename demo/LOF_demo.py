import sys
import logging
from pyod.models.lof import LOF

sys.path.append("../")

from common.dataloader import load_dataset
from common.evaluation import evaluator
from common.utils import pprint
from common.evaluation import evaluate_all

dataset = "SMD"
subdataset = "machine-1-1"
n_neighbors = 20
leaf_size = 30
p = 2   # Parameter for the Minkowski
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
    od = LOF(n_neighbors=n_neighbors, leaf_size=leaf_size, p=p)
    od.fit(x_train)

    # get outlier scores
    anomaly_score = od.decision_function(x_test)

    anomaly_label = x_test_labels

    # Make evaluation
    evaluate_all(anomaly_score, anomaly_label)
