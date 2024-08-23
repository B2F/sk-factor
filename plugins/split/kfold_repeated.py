from sklearn.model_selection import RepeatedKFold
from plugins.split.base_cv import BaseCv

class KfoldRepeated(BaseCv):

    def split(self):

        # Won't work with cross_val_predict (e.g confusion matrix)
        k = RepeatedKFold(n_splits = self._nSplits, n_repeats=self._n_repeats)
        return k.split(self._x, self._y)
