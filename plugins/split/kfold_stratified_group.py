from sklearn.model_selection import StratifiedGroupKFold
from plugins.split.base_cv import BaseCv

class KfoldStratifiedGroup(BaseCv):

    def split(self):
        k = StratifiedGroupKFold(n_splits = self._nSplits)
        return k.split(self._x, self._y, self._groups)
