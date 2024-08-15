from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import KBinsDiscretizer as KBinsDiscretizerBase

class KbinsDiscretizer(BaseTransformer):

    _name = 'k_bins_discretizer'

    def estimator(
        self,
        n_bins = 5,
        encode = 'onehot',
        strategy = 'quantile',
        dtype = None,
        subsample = 200_000,
        random_state = None
    ):

        # Preprocess for columns selected in the .ini config.
        return KBinsDiscretizerBase(
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            dtype=dtype,
            subsample=subsample,
            random_state=random_state
        )
