import os
import numpy as np
import pywt
import matplotlib.image as img
import matplotlib.pyplot as plt
from math import ceil
from sklearn.ensemble import IsolationForest

pattern_swap_change_halfway = './data/pattern_swap/change_halfway'
bitrate_fluctuation_change_halfway = './data/bitrate_fluctuation/change_halfway'
sum_diff_change_halfway = './data/sum_diff/change_halfway'
pattern_swap_change_three_quarters = './data/pattern_swap/change_three_quarters'
bitrate_fluctuation_change_three_quarters = './data/bitrate_fluctuation/change_three_quarters'
sum_diff_change_three_quarters = './data/sum_diff/change_three_quarters'


def readfile(folder_path):
    data = []
    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            with open(file_path, 'r') as infile:
                buf = [file_path, list(map(float, infile.read().splitlines()))]
                data.append(buf)
    return data


def divide_chunks(array, n):
    size = ceil(len(array) / n)
    return list(
        map(lambda x: array[x * size:x * size + size],
        list(range(n)))
    )


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
        anomaly_scores.append(model.decision_function(data))
    return anomaly_scores


def Run_model(name, path, con):
    print('--- ' + name + ' ---')
    data = readfile(path)

    print('generate figures')
    GenerateDWTFigures(path, data)

    print('generate test and train data')
    train, test = divide_chunks(data, 2)
    train_transformed = transform_data(train)
    test_transformed = transform_data(test)
    print('train model and detect anomaly')
    model = train_models(con, train_transformed)
    print(anomaly_detect(model, test_transformed))


Run_model('pattern_swap_change_halfway', pattern_swap_change_halfway, 0.5)
Run_model('bitrate_fluctuation_change_halfway', bitrate_fluctuation_change_halfway, 0.5)
Run_model('sum_diff_change_halfway', sum_diff_change_halfway, 0.5)
Run_model('pattern_swap_change_three_quarters', pattern_swap_change_three_quarters, 0.3)
Run_model('bitrate_fluctuation_change_three_quarters', bitrate_fluctuation_change_three_quarters, 0.3)
Run_model('sum_diff_change_three_quarters', sum_diff_change_three_quarters, 0.3)
