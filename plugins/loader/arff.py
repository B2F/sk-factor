import pandas as pd
import arff as arff
from plugins.loader.base_loader import BaseLoader

class Arff(BaseLoader):

    def _load(self):

        if len(self._files) > 1:
            raise Exception('Cannot load more than one openml dataset at once')

        arff_data = arff.load(open(self._files[0], 'r'))
        
        # Extract column names from attributes (which are tuples of (name, type))
        column_names = [attr[0] for attr in arff_data['attributes']]
        
        # Create DataFrame with all data
        df = pd.DataFrame(data=arff_data['data'], columns=column_names)
        
        return df
