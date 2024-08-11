from sktime.classification.interval_based import TimeSeriesForestClassifier as BaseTSFC
from plugins.pipeline.base_estimator import BaseEstimator
from sktime.datatypes import check_raise
from sktime.datasets import load_covid_3month

class TimeSeriesForestClassifier(BaseEstimator):

    _type = ('classifier')

    def getEstimator(self) -> tuple:

        print('check raise')
        # X, y = load_covid_3month(return_X_y=True)
        # print(X)
        # print(check_raise(X, "pd.DataFrame"))
        # print(X)

        print('check raise 2')
        print(type(self._x))
        print(check_raise(self._x, "pd.DataFrame"))
        exit()
        tsfc = BaseTSFC()
        return ('tsfc', tsfc)
