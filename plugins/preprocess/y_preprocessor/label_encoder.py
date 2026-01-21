from plugins.preprocess.base_y_preprocessor import BaseYPreprocessor
from pathlib import Path
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
import pandas as pd

class LabelEncoder(BaseYPreprocessor):
    """ Generic label encoder for Y (target) variable.
    """

    _id = Path(__file__).stem
    _label_encoder = None

    def preprocess_y(self):
        """ Apply label encoding to the Y variable.

        Returns:
            pd.DataFrame: The encoded Y variable
        """
        if self._arguments is not None and self._arguments is False:
            # If arguments specify not to encode, return original
            return self._y

        # Apply label encoding
        label_col = self._y.columns[0]  # Assuming single column for Y
        le = SklearnLabelEncoder()
        encoded_values = le.fit_transform(self._y.values.flatten())

        # Store the encoder for potential use elsewhere
        self._label_encoder = le

        # Return as DataFrame with same column name
        return pd.DataFrame(encoded_values, columns=[label_col], index=self._y.index)
