from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import FunctionTransformer

class StringToFloat(BaseTransformer):

    def stringToFloat(self, df):
        df = df.map(lambda s: float(s.replace(',', '.')))
        return df

    def transform(self):
        return FunctionTransformer(self.stringToFloat)
