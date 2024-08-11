from imblearn.under_sampling import TomekLinks as BaseTomekLinks
from plugins.pipeline.base_estimator import BaseEstimator

class TomekLinks(BaseEstimator):

    _type = ('sampler')

    def getEstimator(self) -> tuple:

        # Illustrates how a custom config can be used in an estimator class.
        sampling_strategy = eval(self._config['tomek_links']['strategy'])
        nb_jobs = eval(self._config['tomek_links']['nb_jobs'])
        tl = BaseTomekLinks(
            sampling_strategy=list(sampling_strategy),
            n_jobs=nb_jobs
        )
        return ('tomek_links', tl)

