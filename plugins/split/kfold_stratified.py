from sklearn.model_selection import StratifiedKFold
from plugins.split.base_cv import BaseCv

class KfoldStratified(BaseCv):

    def split(self):
        k = StratifiedKFold(n_splits = self._nSplits)
        return k.split(self._x, self._y, self._groups)
