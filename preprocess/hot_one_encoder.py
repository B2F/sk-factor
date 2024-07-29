from preprocess.base_transformer import BaseTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class HotOneEncoder(BaseTransformer):

    def pipeline(self):
        # Preprocess for columns selected in the .ini config.
        return Pipeline(
            steps=[
                ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]
        )
