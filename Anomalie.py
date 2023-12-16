from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pywt
import numpy as np
import sklearn.datasets
import scipy.stats
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError

data = sklearn.datasets.load_files('./data/dataset')
x = []
y = data.target
for item in data.data:
    lines = item.splitlines()
    numbers = []
    for line in lines:
        if len(numbers)<200:
            try:
                numbers.append(float(line))
            except :
                pass
    x.append(numbers)
(
    data_for_train,
    data_for_test,
    label_for_train,
    label_for_test,
) = train_test_split(x, y, train_size=0.7, random_state=31)

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    return [entropy]


def transform_data(data,label):
    list_features = []
    list_unique_labels = list(set(label))
    list_labels = [list_unique_labels.index(elem) for elem in label]
    for x in data:
        list_coeff = pywt.dwt(x, 'haar')
        features = []
        for coeff in list_coeff:
            features += get_features(coeff)
        list_features.append(features)
    return list_features,list_labels

x_train, y_train = transform_data(data_for_train,label_for_train)
x_test, y_test = transform_data(data_for_test,label_for_test)

cls = GradientBoostingClassifier(n_estimators=2000)
cls.fit(x_train, y_train)
train_score = cls.score(x_train, y_train)
test_score = cls.score(x_test, y_test)
print("Train Score for the dataset is about: {}".format(train_score))
print("Test Score for the dataset is about: {}".format(test_score))



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