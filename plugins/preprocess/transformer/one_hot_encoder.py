from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

class OneHotEncoder(BaseTransformer):

    def transform(self):
        # Preprocess for columns selected in the .ini config.
        return SklearnOneHotEncoder(sparse_output=False, handle_unknown='ignore')
