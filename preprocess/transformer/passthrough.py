import pandas as pd
from preprocess.base_transformer import BaseTransformer

class Passthrough(BaseTransformer):

    def transform(self) -> pd.DataFrame:
        return super().transform()
