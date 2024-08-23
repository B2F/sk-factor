from xgboost import XGBRegressor
from plugins.estimators.base_estimator import BaseEstimator

class Xgboost(BaseEstimator):

    _type = 'regressor'

    def getEstimator(self):
        return ('xgboost', XGBRegressor())
