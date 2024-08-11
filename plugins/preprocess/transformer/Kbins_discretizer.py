from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import KBinsDiscretizer as SklearnKBinsDiscretizer

class KbinsDiscretizer(BaseTransformer):

    def transform(self):
        # Preprocess for columns selected in the .ini config.
        return SklearnKBinsDiscretizer(encode="ordinal")
