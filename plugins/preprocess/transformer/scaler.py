from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import StandardScaler

class Scaler(BaseTransformer):

    _name = 'standard_scaler'

    def estimator(self):
        return StandardScaler()
