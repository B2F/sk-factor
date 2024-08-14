from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

class OneHotEncoder(BaseTransformer):

    _name = 'one_hot_encoder'

    def estimator(self):
        # Preprocess for columns selected in the .ini config.
        return SklearnOneHotEncoder(sparse_output=False, handle_unknown='ignore')
