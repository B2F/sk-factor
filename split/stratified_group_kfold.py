from abc import ABC
from sklearn.model_selection import StratifiedGroupKFold

class StratifiedGroupKfold(ABC):

    @staticmethod
    def split(nbSplits, x, y = None, groups = None):
        sgkf = StratifiedGroupKFold(n_splits = nbSplits)
        return sgkf.split(x, y, groups)
