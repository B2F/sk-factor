import pandas as pd
from preprocess.base_transformer import BaseTransformer

class Passthrough(BaseTransformer):

    _action = 'passthrough'

    def transform(self, df) -> pd.DataFrame:
        return super().transform(df)
