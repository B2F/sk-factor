from sklearn.model_selection import KFold
from plugins.split.base_cv import BaseCv

class KfoldShuffle(BaseCv):

    def split(self):
        k = KFold(n_splits = self._nSplits, random_state=self._random_state, shuffle=True)
        return k.split(self._x, self._y)
