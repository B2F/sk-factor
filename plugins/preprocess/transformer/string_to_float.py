from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import FunctionTransformer

class StringToFloat(BaseTransformer):

    _name = 'string_to_float'

    def stringToFloat(self, df):
        df = df.map(lambda s: float(s.replace(',', '.')))
        return df

    def estimator(self):
        return FunctionTransformer(self.stringToFloat)
