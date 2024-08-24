from sklearn.linear_model import SGDClassifier
from plugins.estimators.base_estimator import BaseEstimator

class SgdClassifier(BaseEstimator):

    _type = 'classifier'

    def getEstimator(self):
        return ('sgd_classifier', SGDClassifier(
            loss="modified_huber",
            penalty="elasticnet",
            random_state = self._config.get('training', 'seed'),
            max_iter=50,
        ))
