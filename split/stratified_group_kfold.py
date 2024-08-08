from sklearn.model_selection import StratifiedGroupKFold
from split.base_cv import BaseCv

class StratifiedGroupKfold(BaseCv):

    def split(self):
        sgkf = StratifiedGroupKFold(n_splits = self._nSplits)
        return sgkf.split(self._x, self._y, self._groups)
