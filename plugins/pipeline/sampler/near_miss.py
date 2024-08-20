from imblearn.under_sampling import NearMiss as BaseNearMiss
from plugins.pipeline.base_estimator import BaseEstimator

class NearMiss(BaseEstimator):

    _type = 'sampler'

    def getEstimator(self) -> tuple:

        version = self._config.get('sampling', 'near_miss_version', 2)
        nb_jobs = self._config.get('sampling', 'nb_jobs')
        nm = BaseNearMiss(
            version=version,
            n_jobs=nb_jobs
        )
        return ('near_miss', nm)
