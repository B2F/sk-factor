from abc import ABC
from sklearn.model_selection import GroupKFold

# Base Cross Validator.
class BaseCv(ABC):

    @staticmethod
    def split(nbSplits, x, y = None, groups = None):
        kf = GroupKFold(n_splits=nbSplits)
        return kf.split(x, y, groups)
