from sklearn.model_selection import ShuffleSplit
from plugins.split.base_cv import BaseCv

class Shuffle(BaseCv):

    def split(self):

        s = ShuffleSplit(
            n_splits = self._nSplits,
            test_size=self._test_size,
            random_state=self._random_state
        )

        return s.split(self._x, self._y, self._groups)
