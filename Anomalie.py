from tkinter import X
from turtle import mode
import numpy as np
import sklearn.datasets
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pywt
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError

data_1 = np.random.normal(0, 1, (100, 2))  # Pierwszy zbiór danych
data_2 = np.random.normal(1, 1, (100, 2))  # Drugi zbiór danych z przesunięciem w stosunku do pierwszego


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



data = pywt.dwt(X_train, 'db1')
model.fit(data[1])

test_data = pywt.dwt(X_test,'db1')
print(model.predict(test_data[1]))
