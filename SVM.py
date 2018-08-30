from BakSys import BakardjianSystem

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

class SVM(object):

    """
    Classifier that uses Support Vector Machine algorithm to classify EEG data,
    by analysing it's frequency features.
    """

    def __init__(self,gamma=1,C=1):
        self.self = self
        self.gamma = gamma
        self.C = C
        self.svm = SVC(gamma=self.gamma,C=self.C)
        self.pipe = Pipeline([('scal',MinMaxScaler()),('svm',self.svm)])


    def fit(self,data,target):
        self.pipe.fit(data,target)
        self.data = data
        self.target = target

    def score(self,X_test,y_test):
        prediction = self.pipe.score(X_test,y_test)
        return prediction
