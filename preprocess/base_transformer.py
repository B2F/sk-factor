import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

class BaseTransformer():

    def transformer(self, df) -> pd.DataFrame:
        # Override this method to alter df directly via a FunctionTransformer.
        return df

    def pipeline(self) -> Pipeline:
        # Preprocess for columns selected in the .ini config.
        return Pipeline(
            steps=[
                ('transformer', FunctionTransformer(self.transformer))
            ]
        )
