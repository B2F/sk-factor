from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.impute import SimpleImputer as SimpleImputerBase
import numpy as np

class SimpleImputer(BaseTransformer):

    _name = 'simple_imputer'

    def estimator(
        self,
        missing_values = np.nan,
        strategy = 'mean',
        fill_value = None,
        copy = True,
        add_indicator = False,
        keep_empty_features = False
      ):

        return SimpleImputerBase(
            missing_values = missing_values,
            strategy = strategy,
            fill_value = fill_value,
            copy = copy,
            add_indicator = add_indicator,
            keep_empty_features = keep_empty_features
        )
