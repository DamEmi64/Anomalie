'''
import os
import numpy as np
import pywt
import ADwin
from sklearn.model_selection import train_test_split
import matplotlib.image as img
import matplotlib.pyplot as plt
from math import ceil
import sklearn.datasets
import sklearn
import random
from sklearn.ensemble import IsolationForest
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError

pattern_swap_change_halfway = './data/with_concept/pattern_swap/change_halfway'
bitrate_fluctuation_change_halfway = './data/with_concept/bitrate_fluctuation/change_halfway'
sum_diff_change_halfway = './data/with_concept/sum_diff/change_halfway'
pattern_swap_change_three_quarters = './data/with_concept/pattern_swap/change_three_quarters'
bitrate_fluctuation_change_three_quarters = './data/with_concept/bitrate_fluctuation/change_three_quarters'
sum_diff_change_three_quarters = './data/with_concept/sum_diff/change_three_quarters'
no_concept_int9 = './data/no_concept/int9'


def main():
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
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(x, y, train_size=0.7, random_state=31)
    model = IsolationForest(contamination=0.5)
    model.fit(X=X_train,y=y_train)
    metric = PrequentialError(alpha=1.0)
    
    config = DDMConfig(
        warning_level=2.0,
        drift_level=3.0,
        min_num_instances=25,  # minimum number of instances before checking for concept drift
    )
    detector = DDM(config=config)

    for i, (X, y) in enumerate(zip(X_test, y_test)):
        y_pred = model.predict(np.array().reshape(1, -1))
        error = 1 - (y_pred.item() == y.item())
        metric_error = metric(error_value=error)
        _ = detector.update(value=error)
        status = detector.status
        if status["drift"] and not drift_flag:
            drift_flag = True
            print(f"Concept drift detected at step {i}. Accuracy: {1 - metric_error:.4f}")
    if not drift_flag:
        print("No concept drift detected")
    print(f"Final accuracy: {1 - metric_error:.4f}\n")
    
def stream_test(X_test, y_test, y, metric, detector):
    """Simulate data stream over X_test and y_test. y is the true label."""
    drift_flag = False
    for i, (X, y) in enumerate(zip(X_test, y_test)):
        y_pred = pipeline.predict(X.reshape(1, -1))
        error = 1 - (y_pred.item() == y.item())
        metric_error = metric(error_value=error)
        _ = detector.update(value=error)
        status = detector.status
        if status["drift"] and not drift_flag:
            drift_flag = True
            print(f"Concept drift detected at step {i}. Accuracy: {1 - metric_error:.4f}")
    if not drift_flag:
        print("No concept drift detected")
    print(f"Final accuracy: {1 - metric_error:.4f}\n")


def readfile(folder_path_concept,folder_path_no_concept):
    return 
    


def DWT(data):
    coeffs = pywt.dwt(data, 'haar')
    return coeffs


def GenerateDWTFigures(path, data_files):
    if not os.path.isdir('./images' + path):
        os.makedirs('./images' + path)
    i = 0

    for x, data in data_files:
        coeffs = DWT(data)
        plt.xlabel('sample')
        plt.ylabel('cD')
        plt.plot(coeffs[1])
        plt.title(str(i))
        plt.savefig(os.path.abspath('./images' + path + '/' + str(i)))
        plt.clf()
        i += 1


def transform_data(array_of_data):
    models = []
    for x, data in array_of_data:
        models.append(np.array(data).reshape(-1,1))

    return models


def train_models(con, data):
    model = IsolationForest(contamination=con)
    for simple_data in data:
        model.fit(simple_data)

    return model

def anomaly_detect(model, test_data):
    anomaly_scores = []
    for data in test_data:
        anomaly_scores.append(model.predict(data))
    return anomaly_scores


def Run_model(name, path_concept,path_no_concept, con = 0.5):
    print('--- ' + name + ' ---')
    X,y = readfile(path_concept,path_no_concept)
    print('generate figures')
    #GenerateDWTFigures(path, data)

    print('generate test and train data')
    # Split train (70%) and test (30%)

    print('train model and detect anomaly')
    model = train_models(con, train_transformed)
    print(anomaly_detect(model, test_transformed))


main()
'''