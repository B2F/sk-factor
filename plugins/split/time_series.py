from sklearn.model_selection import TimeSeriesSplit
from plugins.split.base_cv import BaseCv

class TimeSeries(BaseCv):

    def split(self):

        gap = 7
        if self._config.get('split', 'split_gap'):
            gap = self._config.get('split', 'split_gap')

        ts = TimeSeriesSplit(n_splits = self._nSplits, test_size=self._test_size, gap = gap)

        return ts.split(self._x, self._y)
