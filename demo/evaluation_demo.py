import sys

sys.path.append("../")

import numpy as np
from common.evaluation import evaluate_all
from common.utils import pprint

if __name__ == "__main__":
    num_points = 28479
    # anomaly labels for every testing observations
    anomaly_label = np.random.choice([0, 1], size=num_points)

    # anomaly scores for every testing observations
    anomaly_score = np.random.uniform(0, 1, size=num_points)

    # (optional) anomaly score for every training observations
    # if not given, EVT based threshold will not be calculated
    anomaly_score_train = np.random.uniform(0, 1, size=num_points)

    # if anomaly_score_train is given, EVT based threshold will be calculated
    metrics_iter, metrics_evt, theta_iter, theta_evt = evaluate_all(
        anomaly_score, anomaly_label, anomaly_score_train
    )
