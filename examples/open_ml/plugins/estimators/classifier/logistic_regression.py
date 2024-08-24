from sklearn.linear_model import LogisticRegression as LogisticRegressionBase
from plugins.estimators.base_estimator import BaseEstimator

class LogisticRegression(BaseEstimator):

    _type = 'classifier'

    def getEstimator(self):
        return ('logistic_regression', LogisticRegressionBase(
            fit_intercept=True,
            class_weight=self._classWeights,
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5
        ))
