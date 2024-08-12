from sklearn.model_selection import RepeatedKFold
from plugins.split.base_cv import BaseCv

class KfoldRepeated(BaseCv):

    def split(self):

        n_repeats=2
        if 'split' in self._config:
            n_repeats = self._config.get('split', 'n_repeats')

        k = RepeatedKFold(n_splits = self._nSplits, n_repeats=n_repeats)
        return k.split(self._x, self._y, self._groups)
