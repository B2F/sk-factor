from sktime.transformations.series.detrend import Deseasonalizer as BaseDeseasonalizer
from plugins.estimators.base_estimator import BaseEstimator

class Deseasonalizer(BaseEstimator):

    _type = 'transformer'

    def getEstimator(self) -> tuple:

        d = BaseDeseasonalizer()
        return ('deseasonalizer', d)
