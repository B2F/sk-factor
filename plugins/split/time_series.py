from sklearn.model_selection import TimeSeriesSplit
from plugins.split.base_cv import BaseCv

class TimeSeries(BaseCv):

    def split(self):

        gap = 7
        if 'split' in self._config:
            gap = self._config['split']['split_gap']
        ts = TimeSeriesSplit(n_splits = self._nSplits, gap = gap)

        return ts.split(self._x, self._y, self._groups)
