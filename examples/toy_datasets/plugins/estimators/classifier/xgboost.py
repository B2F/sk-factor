from xgboost import XGBClassifier
from plugins.estimators.base_estimator import BaseEstimator

class Xgboost(BaseEstimator):

    _type = 'classifier'

    def getEstimator(self):
        return ('xgboost', XGBClassifier(
            random_state = self._config.get('training', 'seed'),
        ))
