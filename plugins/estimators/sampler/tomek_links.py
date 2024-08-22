from imblearn.under_sampling import TomekLinks as BaseTomekLinks
from plugins.estimators.base_estimator import BaseEstimator

class TomekLinks(BaseEstimator):

    _type = 'sampler'

    def getEstimator(self) -> tuple:

        sampling_strategy = self._config.get('sampling', 'strategy', [])
        nb_jobs = self._config.get('sampling', 'nb_jobs')
        tl = BaseTomekLinks(
            sampling_strategy=list(sampling_strategy),
            n_jobs=nb_jobs
        )
        return ('tomek_links', tl)
