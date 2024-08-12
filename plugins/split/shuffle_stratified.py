from sklearn.model_selection import StratifiedShuffleSplit
from plugins.split.base_cv import BaseCv

class ShuffleStratified(BaseCv):

    def split(self):

        test_size=0.2
        random_state=0

        if self._config.get('split', 'test_size'):
            test_size = self._config.get('split', 'test_size')
        if self._config.get('split', 'random_state'):
            random_state = self._config.get('split', 'random_state')

        s = StratifiedShuffleSplit(n_splits = self._nSplits, test_size=test_size, random_state=random_state)

        return s.split(self._x, self._y, self._groups)
