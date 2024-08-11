from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import KBinsDiscretizer as SklearnKBinsDiscretizer

class KbinsDiscretizer5(BaseTransformer):

    def transform(self):
        return SklearnKBinsDiscretizer(n_bins=5, encode="ordinal", strategy="kmeans")
