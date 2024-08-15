from abc import ABC, abstractmethod
from sklearn.model_selection import GroupKFold
import pandas as pd
from src.engine.config import Config

# Base Cross Validator.
class BaseCv(ABC):

    _config = Config
    _x = pd.DataFrame
    _y = pd.DataFrame
    _nSplits = int
    _groups = str
    _random_state = int
    _test_size = float
    _n_repeats = int

    def __init__(self, config, x, y, n_splits = 5):

        self._config = config
        self._x = x
        self._y = y
        self._nSplits = n_splits

        group = config.get('training', 'splitting_group')
        if group is not None and group in x.columns:
            self._groups = self._x[group]

        self._random_state = config.get('training', 'splitting_random_state')
        self._test_size = config.get('training', 'splitting_test_size')
        self._n_repeats = config.get('training', 'splitting_n_repeats')

    @abstractmethod
    def split(self):
        # Example with GroupKFold:
        kf = GroupKFold(n_splits=self._nSplits)
        return kf.split(self._x, self._y, self._groups)
