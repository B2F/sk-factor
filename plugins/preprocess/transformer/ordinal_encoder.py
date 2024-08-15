from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import OrdinalEncoder as OrdinalEncoderBase
import numpy as np

class OrdinalEncoder(BaseTransformer):

    _name = 'ordinal_encoder'

    def estimator(
        self,
        categories='auto',
        dtype=np.float64,
        handle_unknown='error',
        unknown_value=None,
        encoded_missing_value=np.nan,
        min_frequency=None,
        max_categories=None,
    ):

        # Preprocess for columns selected in the .ini config.
        return OrdinalEncoderBase (
            categories=categories,
            dtype=dtype,
            handle_unknown=handle_unknown,
            unknown_value=unknown_value,
            encoded_missing_value=encoded_missing_value,
            min_frequency=min_frequency,
            max_categories=max_categories
        )
