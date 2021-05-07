import sys
from pyod.models.auto_encoder import AutoEncoder

sys.path.append("../")

from common.dataloader import load_dataset
from common.utils import pprint
from common.evaluation import evaluate_all

dataset = "SMD"
subdataset = "machine-1-1"

hidden_neurons = [64, 32, 32, 64]
batch_size = 32
epochs = 1
l2_regularizer = 0.1
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

    # data preprocessing for MSCRED
    od = AutoEncoder(
        hidden_neurons=hidden_neurons,
        batch_size=batch_size,
        epochs=epochs,
        l2_regularizer=l2_regularizer,
        verbose=1,
    )
    od.fit(x_train)

    # get outlier scores
    anomaly_score = od.decision_function(x_test)

    anomaly_label = x_test_labels

    evaluate_all(anomaly_score, anomaly_label)