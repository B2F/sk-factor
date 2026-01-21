from plugins.preprocess.base_y_preprocessor import BaseYPreprocessor
from pathlib import Path
import pandas as pd

class BinaryRevenue(BaseYPreprocessor):
    """ Binary revenue encoder for adult dataset - converts income levels to binary categories.
    """

    _id = Path(__file__).stem

    def preprocess_y(self):
        """ Convert adult dataset revenue to binary categories (>50K and <=50K).
        
        Returns:
            pd.DataFrame: The binary encoded Y variable
        """
        # Get the label column name
        label_col = self._y.columns[0]  # Assuming single column for Y
        
        # Create binary encoding: 1 for '>50K', 0 for '<=50K'
        def convert_revenue(value):
            if '>50K' in str(value) or value == 1 or value == '1':
                return 1
            else:
                return 0

        # Apply the conversion
        binary_values = self._y.iloc[:, 0].apply(convert_revenue)
        
        # Return as DataFrame with same column name
        return pd.DataFrame(binary_values, columns=[label_col], index=self._y.index)
