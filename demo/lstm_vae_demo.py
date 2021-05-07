import os
import sys

sys.path.append("../")
from networks.lstm_vae import LSTM_Var_Autoencoder
from common.data_preprocess import generate_windows, preprocessor
from common.dataloader import load_dataset
from common.utils import pprint
import tensorflow as tf
from common.evaluation import evaluate_all

dataset = "SMD"
subdataset = "machine-1-1"
window_size = 32
stride = 5
intermediate_dim = 64
z_dim = 3
stateful = True
learning_rate = 0.001
batch_size = 64
num_epochs = 10
point_adjustment = True
iterate_threshold = True

if __name__ == "__main__":
    # load dataset
    data_dict = load_dataset(
        dataset,
        subdataset,
        "all",
    )

    # preprocessing
    pp = preprocessor()
    data_dict = pp.normalize(data_dict)

    # generate sliding windows
    window_dict = generate_windows(data_dict, window_size=window_size, stride=stride)

    train = window_dict["train_windows"]
    test = window_dict["test_windows"]
    test_labels = window_dict["test_labels"]

    # reshape to fitin lstm_vae (not all models need this)
    df_train = train.reshape(-1, *train.shape[-2:])
    df_test = test.reshape(-1, *test.shape[-2:])

    vae = LSTM_Var_Autoencoder(
        intermediate_dim=intermediate_dim,
        z_dim=z_dim,
        n_dim=df_train.shape[-1],
        stateful=stateful,
    )  # default stateful = False

    # Training
    vae.fit(
        df_train,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        opt=tf.train.AdamOptimizer,
        REG_LAMBDA=0.01,
        grad_clip_norm=2,
        optimizer_params=None,
        verbose=True,
    )

    # predict anomaly score for each window
    anomaly_score = vae.predict_prob(df_test).mean(axis=-1)[:, -1]  # mean for all dims
    anomaly_label = window_dict["test_labels"][:, -1]  # last point of each window

    # Make evaluation
    evaluate_all(anomaly_score, anomaly_label)
