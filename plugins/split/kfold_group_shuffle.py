from sklearn.model_selection import GroupShuffleSplit
from plugins.split.base_cv import BaseCv

class KfoldGroupShuffle(BaseCv):

    def split(self):

        k = GroupShuffleSplit(
            n_splits = self._nSplits,
            random_state = self._random_state,
            test_size = self._test_size
        )

        return k.split(self._x, self._y, self._groups)
