from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import KBinsDiscretizer as SklearnKBinsDiscretizer

class KbinsDiscretizer(BaseTransformer):

    _name = 'k_bins_discretizer'

    def estimator(self, n_bins = 5, strategy = 'quantile'):

        # Preprocess for columns selected in the .ini config.
        return SklearnKBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy = strategy)
