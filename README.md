## Multivariate KPIs Anomaly Detection Benchmark

**This repository is a multivariate KPIs anomaly detection toolkit with a comprehensive benchmark protocol. Our goal is to make a software system more reliable via AIOps.**

Key Performance Indicators (KPIs) are recorded for continuously monitoring a system in different aspects, for example, CPU utilization, service response delay, network traffic, etc. Especally, multivariate KPIs are a group of correlated univariate (single) KPIs used to monitor different aspects of an entity.

This repository is a benchmark for state-of-the-art anomaly detection methods in the literature over multivaraite Key Performance Indicators. Specifically, a comprehensive evaluation protocal is integraed, with this protocal, one can easily evaluate his own multivariate anomaly detection method and compare with existing works.

### Our evaluation protocol

Our evaluation protocol consists of following four metrics to evaluate an anomaly detection method.

- **Accuracy** (Precision, Recall, F1-score): *how accruate a model is?*

  - with **or** without point adjustment
  - select a threshold via iteration **or** extreme value theory (EVT)

- **Salience**: *how much can a model highlight an anomaly?*

  ![Salience computation](docs/imgs/salience_computation.png)

- **Delay**: *how timely can a model report an anomaly?*

- **Efficiency**: *how fast can a model be trained and perform anomaly detection?*

### Datasets 

The following datasets are kindly released by different institutions or schools. Raw datasets could be downloaded or applied from the link right behind the dataset names. The processed datasets can be found [here](https://drive.google.com/drive/folders/1NEGyB4y8CvUB8TX2Wh83Eas_QHtufGPR?usp=sharing)⬇️ (SMD, SMAP, and MSL).

- Server Machine Datase (**SMD**) [Download raw datasets⬇️](https://github.com/NetManAIOps/OmniAnomaly.git)

  > Collected from a large Internet company containing a 5-week-long monitoring KPIs of 28 machines. The meaning for each KPI could be found [here](https://github.com/NetManAIOps/OmniAnomaly/issues/22).

- Soil Moisture Active Passive satellite (**SMAP**) and Mars Science Laboratory rovel (**MSL**) [Download raw datasets⬇️](link)

  > They are collected from the running spacecraft and contain a set of telemetry anomalies corresponding to actual spacecraft issues involving various subsystems and channel types.

- Secure Water Treatment (**WADI**) [Apply here\*](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

  >WADI is collected from a real-world industrial water treatment plant, which contains 11-day-long multivariate KPIs. Particularly, the system is in a normal state in the first seven days and is under attack in the following four days.

- Water Distribution (**SWAT**) [Apply here\*](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

  > An extended dataset of SWAT. 14-day-long operation KPIs are collected when the system is running normally and 2-day-long KPIs are obtained when the system is in attack scenarios.

\* WADI and SWAT datasets were released by iTrust, which should be individually applied. One can request the raw datasets and preprocess them with our preprocessing scripts.

### Requirement

> git clone https://github.com/ase21-843/code
>
> cd ./code
>
> pip install -r requirements.txt  

### Run our demo

```
cd demo
python lstm_demo.py
```

### API Usage

```python
# 1. load datasets
# keys in data_dict: 
# "train": numpy array with shape "num_training_points x num_kpis"
# "test": numpy array with shape "num_testing_points x num_kpis"
# "test_labels": numpy array with shape "num_testing_points x 1"
data_dict = load_dataset(dataset, subdataset)

# 2. normalize data
pp = data_preprocess.preprocessor()
data_dict = pp.normalize(data_dict, method="minmax")

# 3. generate sliding windows
# keys in window_dict: 
# "train_windows": numpy array with shape "num_training_windows x window_size x num_kpis"
# "test_windows": numpy array with shape "num_testing_windows x window_size x num_kpis"
# "test_labels": numpy array with shape "num_testing_windows x window_size x 1"
window_dict = data_preprocess.generate_windows(
  data_dict,
  window_size=window_size,
  stride=stride,
)

# 4. batch windows of all data (optional) 
train_iterator = WindowIterator(...)
test_iterator = WindowIterator(...)
    
# 5. initialize a model (e.g., LSTM)
encoder = LSTM(...)

# 6. train a model
encoder.fit(...)

# 7. detect anomlies on test data (inference)
records = encoder.predict_prob(...)
anomaly_score = records["score"] # num_testing_points x 1
anomaly_label = records["anomaly_label"] #num_testing_points x 1

# 8. evaluation
evaluate_all(anomaly_score, anomaly_label)

​```
Evaluation result sample:
1.AUC
        0.864
2.Salience
        0.4181
3.Iteration Based
        adj_f1
                0.9335
        adj_precision
                0.8752
        adj_recall
                1.0
        delay
                1
        f1
                0.3821
        precision
                0.6544
        recall
                0.2699
        theta
                0.0101
​```
```

### Evaluate a user-defined method

We have integrated the overall evaluation protocol within an easy-to-use function, which could be used as shown in the [demo/evaluation_demo.py](https://github.com/ase21-843/code/blob/main/demo/evaluation_demo.py).  If ``anomaly_score`` and ``anomaly_label`` are provided, thresholds iteration could be conducted. In addition, if ``anomaly_score_train`` is given (optional), a threshold can be automatically selected via EVT.

```python
num_points = 65535
# anomaly labels for every testing observations
anomaly_label = np.random.choice([0, 1], size=num_points)

# anomaly scores for every testing observations
anomaly_score = np.random.uniform(0, 1, size=num_points)

# (optional) anomaly score for every training observations
# if not given, EVT based threshold will not be calculated
anomaly_score_train = np.random.uniform(0, 1, size=num_points)

# if anomaly_score_train is given, EVT based threshold will be calculated
metrics_iter, metrics_evt, theta_iter, theta_evt = evaluate_all(anomaly_score, anomaly_label, anomaly_score_train)

​```
Evaluation result sample:
1.AUC
        0.5038
2.Salience
        0.1671
3.Iteration Based
        adj_f1
                0.6652
        adj_precision
                0.4983
        adj_recall
                1.0
        delay
                0
        f1
                0.6652
        precision
                0.4983
        recall
                1.0
        theta
                0.0
4.EVT Based
        adj_f1
                0.0054
        adj_precision
                0.6496
        adj_recall
                0.0027
        delay
                32
        f1
                0.0021
        precision
                0.4146
        recall
                0.001
        theta
                0.001
​```
```

### Models

**General Machine Learning-based Models**

| Model   | Paper reference                                              |
| :------ | :----------------------------------------------------------- |
| KNN     | **[SIGMOD'2000]** Sridhar Ramaswamy, Rajeev Rastogi, Kyuseok Shim. Efficient Algorithms for Mining Outliers from Large Data Sets |
| LOF     | **[SIGMOD'2000]** Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, Jörg Sander LOF: Identifying Density-Based Local Outliers |
| PCA     | **[2003]** Shyu M L, Chen S C, Sarinnapakorn K, et al. A novel anomaly detection scheme based on principal component classifier |
| iForest | **[ICDM'2008]** Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou: Isolation Forest |
| LODA    | **[Machine Learning'2016]** Tomás Pevný. Loda**:** Lightweight online detector of anomalies |

**General Machine Learning-based Models**

| Model       | Paper reference                                              |
| :---------- | :----------------------------------------------------------- |
| AE          | **[AAAI'2019]** Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, Luca Benini. Anomaly Detection Using Autoencoders in High Performance Computing Systems |
| LSTM        | **[KDD'2018]** Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, Tom Söderström. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding |
| LSTM-VAE    | **[Arxiv'2017]** A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder |
| DAGMM       | **[ICLR'2018]** Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Dae-ki Cho, Haifeng Chen. Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection |
| MAD-GAN     | **[ICANN'2019]** Dan Li, Dacheng Chen, Baihong Jin, Lei Shi, Jonathan Goh, See-Kiong Ng. MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks |
| MSCRED      | **[AAAI'19]** Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, Nitesh V. Chawla. A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data. |
| OmniAnomaly | **[KDD'2019]** Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, Dan Pei: Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network |



### Acknowledge

We thanks the following repositories for their open-sourced implementations, which are unified and re-evaluated in our paper.

- Pyod
- LSTM-VAE
- DAGMM
- MAD-GAN
- MSCRED
- OmniAnomaly