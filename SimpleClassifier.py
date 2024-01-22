from collections import Counter
from queue import Full
from site import USER_BASE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pywt
import math
import numpy as np
import sklearn.datasets
import scipy.stats
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError

class SimpleClassifier:
   
    def get_features(self,list_values):
        return [scipy.stats.entropy(list_values)] + [np.mean(list_values)] + [np.std(list_values)]

    def transform_data(self,data,label):
        list_features = []
        list_unique_labels = list(set(label))
        list_labels = [list_unique_labels.index(elem) for elem in label]
        for x in data:
            list_coeff = pywt.dwt(x, 'haar')
            features = []
            for coeff in list_coeff:
                features += self.get_features(coeff)
            features = [v for v in features if not math.isinf(v)]
            list_features.append(features)
        return list_features,list_labels
    
    def run(self, x, y, coma):
        train_scores = []
        test_scores = []
        result = 'SimpleClassifier' + coma + coma + coma + coma + '\n'
        result += 'Lp' + coma + 'train score' + coma + 'test score\n'
        (
            data_for_train,
            data_for_test,
            label_for_train,
            label_for_test,
        ) = train_test_split(x, y, train_size=0.7)       
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        for i in range(2):
            x_train, y_train = self.transform_data(data_for_train, label_for_train)
            x_test, y_test = self.transform_data(data_for_test, label_for_test)

            cls = GradientBoostingClassifier(n_estimators=2000)
            cls.fit(x_train, y_train)
            train_score = cls.score(x_train, y_train) # always 1.0
            test_score = cls.score(x_test, y_test)
            train_scores.append(train_score) # always 1.0
            test_scores.append(test_score)
            result += str(i) + coma +  str(train_score)+coma+str(test_score)+'\n' 
            print("Train Score for the dataset is about: {}".format(train_score)) # always 1.0
            print("Test Score for the dataset is about: {}".format(test_score))
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        print("Average train Score for the dataset is about: {}".format(np.mean(train_scores))) # always 1.0
        print("Average test Score for the dataset is about: {}".format(np.mean(test_scores)))
        
        file = open('./Result/SimpleClassifier.csv', 'w+')
        file.write(result)
        
        return test_scores
        



