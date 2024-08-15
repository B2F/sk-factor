from plugins.preprocess.base_selector import BaseSelector
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class SelectStrings(BaseSelector):

    def estimator(self, score_func=f_classif, k=10):

        return SelectKBest(score_func=score_func, k=k)
