from sklearn.model_selection import ShuffleSplit
from plugins.split.base_cv import BaseCv

class Shuffle(BaseCv):

    def split(self):

        test_size=0.2
        random_state=0
        if 'split' in self._config:
            test_size = self._config['split']['test_size']
            random_state = self._config['split']['random_state']
        s = ShuffleSplit(n_splits = self._nSplits, test_size=test_size, random_state=random_state)

        return s.split(self._x, self._y, self._groups)
