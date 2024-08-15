from plugins.preprocess.base_selector import BaseSelector
from sklearn.feature_selection import VarianceThreshold as VarianceThresholdBase

class VarianceThreshold(BaseSelector):

    def estimator(self, threshold=0):

        return VarianceThresholdBase(threshold=threshold)
