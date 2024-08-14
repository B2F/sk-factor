from plugins.preprocess.transformer.kbins_discretizer import KbinsDiscretizer
from pathlib import Path

class KbinsDiscretizer10Kmeans(KbinsDiscretizer):

    _name = Path(__file__).stem

    def estimator(self, n_bins = 10, strategy = 'kmeans'):

        return super().estimator(n_bins, strategy)
