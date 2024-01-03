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

while True:
    print("1.Ucz model - wersja prosta")
    print("2.Ucz model - wersja zaawansowana")
    print("3.Ucz model - ograniczone okno")
    print("4.Zamknij program")
    user = input("")
    if user == '1':
        SimpleClassifier().run()
    elif user == '2':
        FullClassifier().run()
    elif user == '3':
        HalfWindowClassifier().run()
        pass
    elif user == '4':
        break