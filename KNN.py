import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


class KNN(object):

    """
    Classifier that uses Support Vector Machine algorithm to classify EEG data,
    by analysing it's frequency features.

    Attributes
    ----------
    n_neighbors : integer
        how many neighbors should be analysed during classyfing

    """

    def __init__(self,n_neighbors=3):
        self.self = self
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(algorithm='kd_tree',n_neighbors=self.n_neighbors)
        self.pipe = Pipeline([('svm',self.knn)])


    def fit(self,data,target):
        self.pipe.fit(data,target)
        self.data = data
        self.target = target

    def score(self,X_test,y_test):
        prediction = self.pipe.score(X_test,y_test)
        return prediction
