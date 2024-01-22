import math
import numpy as np
import pywt
import scipy.stats
import sklearn.datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from SimpleClassifier import SimpleClassifier
from fullClassifier import FullClassifier
from halfWindowClassifier import HalfWindowClassifier

coma_csv = ';'

def readData():
    data = sklearn.datasets.load_files('./data/dataset_half')
    x = []
    y = data.target
    for item in data.data:
        lines = item.splitlines()
        numbers = []
        for line in lines:
                try:
                    numbers.append(float(line))
                except :
                    pass
        x.append(numbers)

    return x, y

def SaveToCSV():
    result = 'Lp' + coma_csv + 'simple - balanced' + coma_csv + 'simple - mean' + coma_csv + 'full - balanced' + coma_csv + 'full - mean' + coma_csv + 'half window - balanced' + coma_csv + 'half window - mean' + '\n'
    for i in range(0,len(results_simple_balanced)):
        result_full = '-'
        result_full_mean = '-'
        if i < len(results_full_balanced):
            result_full = results_full_balanced[i]
            result_full_mean = results_full_mean[i]
        result_half = '-'
        result_half_mean = '-'
        if i < len(results_half_balanced):
            result_half = results_half_balanced[i]
            result_half_mean = results_half_mean[i]
            
        result+=str(i) + coma_csv + str(results_simple_balanced[i]) + coma_csv + str(results_simple_mean[i]) + coma_csv + str(result_full) + coma_csv + str(result_full_mean) + coma_csv + str(result_half) + coma_csv + str(result_half_mean) + '\n'
    file = open('./Result/Results.csv', 'w+')
    file.write(result)

print('Data loading')
dataset_x, dataset_y = readData()
print('Data loaded')
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")   

results_simple_balanced = []
results_full_balanced = []
results_half_balanced = []
results_simple_mean = []
results_full_mean = []
results_half_mean = []

while True:
    print("1.Ucz model - wersja prosta")
    print("2.Ucz model - wersja zaawansowana")
    print("3.Ucz model - ograniczone okno")
    print("4.Ucz wszystkie modele")
    print("5.Zamknij program")
    user = input("")
    if user == '1':
        results_simple_balanced , results_simple_mean = SimpleClassifier().run(dataset_x,dataset_y,coma_csv)
    elif user == '2':
        results_full_balanced, results_full_mean = FullClassifier().run(dataset_x,dataset_y,coma_csv)
    elif user == '3':
        results_half_balanced, results_half_mean = HalfWindowClassifier().run(dataset_x,dataset_y,coma_csv)
        pass
    elif user == '4':
        print('Simple Classifier')
        results_simple_balanced , results_simple_mean = SimpleClassifier().run(dataset_x,dataset_y,coma_csv)
        print('Full Classifier')
        results_full_balanced, results_full_mean = FullClassifier().run(dataset_x,dataset_y,coma_csv)
        print('Half Window Classifier')
        results_half_balanced, results_half_mean = HalfWindowClassifier().run(dataset_x,dataset_y,coma_csv)
        print("Saving data to compare...")
        SaveToCSV()
        print("Data saved")
    elif user == '5':
        SaveToCSV()
        break