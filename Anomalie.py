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
    data = sklearn.datasets.load_files('./data/dataset')
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
    result = 'Lp' + coma_csv + 'simple' + coma_csv + 'full' + coma_csv + 'half window\n'
    for i in range(0,len(results_simple)):
        result_full = '-'
        if i < len(results_full):
            result_full[i]
        result_half = '-'
        if i < len(results_half):
            results_half[i]
        result+=str(i) + coma_csv + str(results_simple[i]) + coma_csv + str(result_full) + coma_csv + str(result_half) + '\n'
    file = open('./Result/Results.csv', 'w+')
    file.write(result)

print('Data loading')
dataset_x, dataset_y = readData()
print('Data loaded')
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")   

results_simple = []
results_half = []
results_full = []

while True:
    print("1.Ucz model - wersja prosta")
    print("2.Ucz model - wersja zaawansowana")
    print("3.Ucz model - ograniczone okno")
    print("4.Zamknij program")
    user = input("")
    if user == '1':
        results_simple = SimpleClassifier().run(dataset_x,dataset_y,coma_csv)
    elif user == '2':
        results_full = FullClassifier().run(dataset_x,dataset_y,coma_csv)
    elif user == '3':
        results_half = HalfWindowClassifier().run(dataset_x,dataset_y,coma_csv)
        pass
    elif user == '4':
        SaveToCSV()
        break