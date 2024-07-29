from preprocess.base_transformer import BaseTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

class OneHotEncoder(BaseTransformer):

    def pipeline(self):
        # Preprocess for columns selected in the .ini config.
        return Pipeline(
            steps=[
                ('one_hot', SklearnOneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]
        )
