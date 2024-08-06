from preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder
from sklearn.pipeline import Pipeline

class OrdinalEncoder(BaseTransformer):

    def pipeline(self):
        # Preprocess for columns selected in the .ini config.
        return Pipeline(
            steps=[
                ('label', SklearnOrdinalEncoder())
            ]
        )
