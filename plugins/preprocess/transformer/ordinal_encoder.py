from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder

class OrdinalEncoder(BaseTransformer):

    _name = 'ordinal_encoder'

    def estimator(self):

        # Preprocess for columns selected in the .ini config.
        return SklearnOrdinalEncoder()
