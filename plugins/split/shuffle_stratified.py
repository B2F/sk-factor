from sklearn.model_selection import StratifiedShuffleSplit
from plugins.split.base_cv import BaseCv

class ShuffleStratified(BaseCv):

    def split(self):

        s = StratifiedShuffleSplit(
            n_splits = self._nSplits,
            test_size=self._test_size,
            random_state=self._random_state)

        return s.split(self._x, self._y)
