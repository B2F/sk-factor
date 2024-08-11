from sklearn.preprocessing import PowerTransformer
from plugins.pipeline.base_estimator import BaseEstimator

class YeoJohnson(BaseEstimator):

    _type = 'transformer'

    def getEstimator(self) -> tuple:

        yj = PowerTransformer()
        return ('yeo_johnson', yj)
