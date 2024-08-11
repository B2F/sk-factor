from sklearn.model_selection import GroupKFold
from plugins.split.base_cv import BaseCv

class KfoldGroup(BaseCv):

    def split(self):
        k = GroupKFold(n_splits = self._nSplits)
        return k.split(self._x, self._y, self._groups)
