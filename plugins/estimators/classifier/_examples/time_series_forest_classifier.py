from sktime.classification.interval_based import TimeSeriesForestClassifier as BaseTSFC
from plugins.estimators.base_estimator import BaseEstimator

class TimeSeriesForestClassifier(BaseEstimator):

    _type = 'classifier'

    def getEstimator(
        self,
        n_estimators = 200,
        min_interval = 3,
        n_jobs = 1,
        inner_series_length = None,
        random_state = None
    ) -> tuple:

        tsfc = BaseTSFC(
            n_estimators = n_estimators,
            min_interval = min_interval,
            n_jobs = n_jobs,
            inner_series_length = inner_series_length,
            random_state = random_state
        )

        return ('tsfc', tsfc)
