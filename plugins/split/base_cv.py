from abc import ABC, abstractmethod
from sklearn.model_selection import GroupKFold
import pandas as pd

# Base Cross Validator.
class BaseCv(ABC):

    _config = dict
    _x = pd.DataFrame
    _y = pd.DataFrame
    _nSplits = int
    _groups = str

    def __init__(self, config, x, y, n_splits = 2, groups = None):
        self._config = config
        self._x = x
        self._y = y
        self._nSplits = n_splits
        self._groups = groups

    @abstractmethod
    def split(self):
        # Example with GroupKFold:
        kf = GroupKFold(n_splits=self._nSplits)
        return kf.split(self._x, self._y, self._groups)
