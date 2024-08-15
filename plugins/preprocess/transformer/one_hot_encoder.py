from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import OneHotEncoder as OneHotEncoderBase
import numpy as np

class OneHotEncoder(BaseTransformer):

    _name = 'one_hot_encoder'

    def estimator(
        self,
        categories='auto',
        drop=None,
        sparse_output=True,
        dtype=np.float64,
        handle_unknown='error',
        min_frequency=None,
        max_categories=None,
        feature_name_combiner='concat'
    ):

        # Preprocess for columns selected in the .ini config.
        return OneHotEncoderBase(
            categories=categories,
            drop=drop,
            sparse_output=sparse_output,
            dtype=dtype,
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
            max_categories=max_categories,
            feature_name_combiner=feature_name_combiner
        )
