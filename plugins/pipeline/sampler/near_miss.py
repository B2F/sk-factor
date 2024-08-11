from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import NearMiss as BaseNearMiss
from plugins.pipeline.base_estimator import BaseEstimator

class NearMiss(BaseEstimator):

    _type = ('sampler')

    def getEstimator(self) -> tuple:

        # Illustrates how a custom config can be used in an estimator class.
        if self._config['sampling'].get('version'):
            version = eval(self._config['sampling']['version'])
        else:
            version = 3
        nm = BaseNearMiss(version=version)
        return ('near_miss', nm)
