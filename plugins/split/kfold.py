from sklearn.model_selection import KFold
from plugins.split.base_cv import BaseCv

class Kfold(BaseCv):

    def split(self):
        k = KFold(n_splits = self._nSplits)
        return k.split(self._x, self._y)
