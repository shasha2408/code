import sys

sys.path.append("../")

from networks.mscred.mscred import MSCRED
from common.dataloader import load_dataset
from common.evaluation import evaluate_all

dataset = "SMD"
subdataset = "machine-1-1"
device = "0"  # cuda:0, a string
step_max = 5
gap_time = 10
win_size = [10, 30, 60]  # sliding window size
in_channels_encoder = 3
in_channels_decoder = 256
save_path = "../mscred_data/" + dataset + "/" + subdataset + "/"
learning_rate = 0.0002
epoch = 1
thred_b = 0.005
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    # load dataset
    data_dict = load_dataset(dataset, subdataset, use_dim="all")

    x_train = data_dict["train"]
    x_test = data_dict["test"]
    x_test_labels = data_dict["test_labels"]

    mscred = MSCRED(
        in_channels_encoder,
        in_channels_decoder,
        save_path,
        device,
        step_max,
        gap_time,
        win_size,
        learning_rate,
        epoch,
        thred_b,
    )

    mscred.fit(data_dict)

    anomaly_score, anomaly_label = mscred.predict_prob(
        len(x_train), x_test, x_test_labels
    )

    # Make evaluation
    evaluate_all(anomaly_score, anomaly_label)
