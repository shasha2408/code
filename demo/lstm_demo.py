import os
import sys

sys.path.append("../")
import json
import logging
import math
import numpy
import pandas
import torch
import pandas as pd
from common import data_preprocess
from common.config import subdatasets
from common.dataloader import load_dataset
from common.batching import WindowIterator
from common.utils import print_to_json, update_from_nni_params, seed_everything, pprint
from networks.lstm import LSTM
from common.evaluation import evaluate_all

seed_everything(2020)

dataset = "SMD"
subdataset = "machine-1-1"
normalize = "minmax"
save_path = "./save_dir"
batch_size = 64
device = -1  # -1 for cpu, 0 for cuda:0
window_size = 32
stride = 5
nb_epoch = 1
patience = 5

lr = 0.001
hidden_size = 64
num_layers = 1
dropout = 0
prediction_length = 1
prediction_dims = []
iterate_threshold = True
point_adjustment = True

if __name__ == "__main__":
    data_dict = load_dataset(
        dataset,
        subdataset,
    )

    pp = data_preprocess.preprocessor()
    data_dict = pp.normalize(data_dict, method=normalize)
    os.makedirs(save_path, exist_ok=True)
    pp.save(save_path)

    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=window_size,
        stride=stride,
    )

    train_iterator = WindowIterator(
        window_dict["train_windows"], batch_size=batch_size, shuffle=True
    )
    test_iterator = WindowIterator(
        window_dict["test_windows"], batch_size=4096, shuffle=False
    )

    print("Proceeding using {}...".format(device))

    encoder = LSTM(
        in_channels=data_dict["dim"],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        window_size=window_size,
        prediction_length=prediction_length,
        prediction_dims=prediction_dims,
        patience=patience,
        save_path=save_path,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        lr=lr,
        device=device,
    )

    encoder.fit(
        train_iterator,
        test_iterator=test_iterator.loader,
        test_labels=window_dict["test_labels"],
    )

    encoder.load_encoder()
    anomaly_score, anomaly_label = encoder.predict_prob(
        test_iterator.loader, window_dict["test_labels"]
    )

    evaluate_all(anomaly_score, anomaly_label)
