from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import OneHotEncoder as OneHotEncoderBase
import numpy as np

class OneHotEncoder(BaseTransformer):

    _name = 'one_hot_encoder'

    def estimator(
        self,
        categories='auto',
        drop=None,
        dtype=np.float64,
        handle_unknown='error',
        min_frequency=None,
        max_categories=None,
        feature_name_combiner='concat'
    ):

        # Preprocess for columns selected in the .ini config.
        return OneHotEncoderBase(
            # The preprocessing fitting requires a dense DataFrame.
            sparse_output=False,
            categories=categories,
            drop=drop,
            dtype=dtype,
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
            max_categories=max_categories,
            feature_name_combiner=feature_name_combiner
        )
