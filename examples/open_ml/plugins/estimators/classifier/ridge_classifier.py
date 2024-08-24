from sklearn.linear_model import RidgeClassifier as RidgeClassifierBase
from plugins.estimators.base_estimator import BaseEstimator

class RidgeClassifier(BaseEstimator):

    _type = 'classifier'

    def getEstimator(self):
        return ('ridge_classifier', RidgeClassifierBase(
            class_weight=self._classWeights,
            solver='sparse_cg',
            fit_intercept=True
        ))
