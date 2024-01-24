from collections import Counter
from queue import Full
from site import USER_BASE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
import pywt
import math
import numpy as np
import sklearn.datasets
import scipy.stats
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError

class FullClassifier:
    
    def get_features(self,list_values):
        return [np.var(list_values)] + [scipy.stats.skew(list_values)] + [scipy.stats.kurtosis(list_values)] + [
            np.sum(np.diff(list_values)[1:] * np.diff(list_values)[:-1] < 0)]


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
 
    def split_data(self,x,y):
        data_for_train = []
        data_for_test = []
        label_for_train = []
        label_for_test = []
        
        rskf = RepeatedStratifiedKFold(n_splits=3, random_state=42)

        for i, (train_index, test_index) in enumerate(rskf.split(x, y)):
            if i == 10:
                for idx in train_index:
                    data_for_train.append(x[idx])
                    label_for_train.append(y[idx])
                for idx in test_index:
                    data_for_test.append(x[idx])
                    label_for_test.append(y[idx])
        return data_for_train,data_for_test,label_for_train,label_for_test    

    def run(self, x, y, coma):
        train_scores = []
        test_scores_balanced = []
        test_scores_mean = []
        result = 'FullClassifier' + coma + coma + coma + coma + '\n'
        result += 'Lp' + coma + 'train score' + coma + 'test score balanced' + coma + 'test score mean\n'
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")    

        for i in range(2):
            #to można potem zakometować
            (
                data_for_train,
                data_for_test,
                label_for_train,
                label_for_test,
            ) = self.split_data(x,y)
            
            x_train, y_train = self.transform_data(data_for_train, label_for_train)
            x_test, y_test = self.transform_data(data_for_test, label_for_test)

            cls = GradientBoostingClassifier(n_estimators=2000)
            cls.fit(x_train, y_train)
            y_result = cls.predict(x_train)
            train_score = balanced_accuracy_score(y_result,y_train)
            print("Train Score for the dataset is about: {}".format(train_score)) # always 1.0
            y_result = cls.predict(x_test)
            test_score = balanced_accuracy_score(y_result,y_test)
            print("Balanced Test Score for the dataset is about: {}".format(test_score))
            
            train_scores.append(train_score) # always 1.0
            test_scores_balanced.append(test_score)
            test_score_mean = cls.score(x_test,y_test)
            test_scores_mean.append(test_score_mean)
            print("Mean Test Score for the dataset is about: {}".format(test_score_mean))
            
            result += str(i) + coma +  str(train_score)+coma+str(test_score)+ coma+str(test_score_mean) +'\n' 
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        print("Average train Score for the dataset is about: {}".format(np.mean(train_scores))) # always 1.0
        print("Average balanced test Score for the dataset is about: {}".format(np.mean(test_scores_balanced)))
        print("Average mean test Score for the dataset is about: {}".format(np.mean(test_scores_mean)))   
        file = open('./Result/FullClassifier.csv', 'w+')
        file.write(result)
        
        return test_scores_balanced, test_scores_mean



