from plugins.preprocess.base_selector import BaseSelector
from sklearn.compose import make_column_selector
import numpy as np

class SelectStrings(BaseSelector):

    def estimator(self):

        return make_column_selector(dtype_include=object)
