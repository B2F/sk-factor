from imblearn.under_sampling import NearMiss as BaseNearMiss
from plugins.estimators.base_estimator import BaseEstimator

class NearMiss(BaseEstimator):

    _type = 'sampler'

    def getEstimator(self) -> tuple:

        version = self._config.get('sampling', 'near_miss_version', 2)
        nm = BaseNearMiss(
            version=version,
        )
        return ('near_miss', nm)
