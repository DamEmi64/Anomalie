import math

import numpy as np
import pywt
import scipy.stats
import sklearn.datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

data = sklearn.datasets.load_files('./data/dataset')

x = []
y = data.target
for item in data.data:
    lines = item.splitlines()
    numbers = []
    for line in lines:
        try:
            numbers.append(float(line))
        except:
            pass
    x.append(numbers)


def get_features(list_values):
    # return [scipy.stats.entropy(list_values)] + [np.mean(list_values)] + [np.std(list_values)]
    return [np.var(list_values)] + [scipy.stats.skew(list_values)] + [scipy.stats.kurtosis(list_values)] + [
        np.sum(np.diff(list_values)[1:] * np.diff(list_values)[:-1] < 0)]


def transform_data(data, label):
    list_features = []
    list_unique_labels = list(set(label))
    list_labels = [list_unique_labels.index(elem) for elem in label]
    for x in data:
        list_coeff = pywt.dwt(x, 'haar')
        features = []
        for coeff in list_coeff:
            features += get_features(coeff)
        features = [v for v in features if not math.isinf(v)]
        list_features.append(features)
    return list_features, list_labels


# train_scores = []
test_scores = []

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
for i in range(10):
    (
        data_for_train,
        data_for_test,
        label_for_train,
        label_for_test,
    ) = train_test_split(x, y, train_size=0.7)
    # ) = train_test_split(x, y, train_size=0.7, random_state=31)

    x_train, y_train = transform_data(data_for_train, label_for_train)
    x_test, y_test = transform_data(data_for_test, label_for_test)

    cls = GradientBoostingClassifier(n_estimators=2000)
    cls.fit(x_train, y_train)
    # train_score = cls.score(x_train, y_train) # always 1.0
    test_score = cls.score(x_test, y_test)
    # train_scores.append(train_score) # always 1.0
    test_scores.append(test_score)
    # print("Train Score for the dataset is about: {}".format(train_score)) # always 1.0
    print("Test Score for the dataset is about: {}".format(test_score))
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# print("Average train Score for the dataset is about: {}".format(np.mean(train_scores))) # always 1.0
print("Average test Score for the dataset is about: {}".format(np.mean(test_scores)))

'''

def get_uci_har_features(dataset, labels, waveletname):
    uci_har_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            list_coeff = pywt.wavedec(signal, waveletname)
            for coeff in list_coeff:
                features += get_features(coeff)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y

def get_ecg_features(ecg_data, ecg_labels, waveletname):
    list_features = []
    list_unique_labels = list(set(ecg_labels))
    list_labels = [list_unique_labels.index(elem) for elem in ecg_labels]
    for signal in ecg_data:
        list_coeff = pywt.wavedec(signal, waveletname)
        features = []
        for coeff in list_coeff:
            features += get_features(coeff)
        list_features.append(features)
    return list_features, list_labels

X_train_ecg, Y_train_ecg = get_ecg_features(train_data_ecg, train_labels_ecg, 'db4')
X_test_ecg, Y_test_ecg = get_ecg_features(test_data_ecg, test_labels_ecg, 'db4')

X_train_ucihar, Y_train_ucihar = get_uci_har_features(train_signals_ucihar, train_labels_ucihar, 'rbio3.1')
X_test_ucihar, Y_test_ucihar = get_uci_har_features(test_signals_ucihar, test_labels_ucihar, 'rbio3.1')
'''
