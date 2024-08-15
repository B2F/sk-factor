from plugins.preprocess.base_selector import BaseSelector
from sklearn.feature_selection import SelectPercentile as SelectPercentileBase
from sklearn.feature_selection import f_classif

class SelectPercentile(BaseSelector):

    def estimator(self, score_func=f_classif, percentile=10):

        return SelectPercentileBase(score_func=score_func, percentile=percentile)
