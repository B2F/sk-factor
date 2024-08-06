from preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder
from sklearn.pipeline import Pipeline

class OrdinalEncoder(BaseTransformer):

    def transform(self):
        # Preprocess for columns selected in the .ini config.
        return SklearnOrdinalEncoder()
