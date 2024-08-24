from plugins.estimators.base_estimator import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier

class KneighborsClassifier(BaseEstimator):

    _type = 'classifier'

    def getEstimator(self):
        return ('kneightbors_classifier', KNeighborsClassifier())
