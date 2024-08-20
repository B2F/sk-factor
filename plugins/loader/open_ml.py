import pandas as pd
from plugins.loader.base_loader import BaseLoader
from sklearn.datasets import fetch_openml

class OpenMl(BaseLoader):

    def _load(self):

        if len(self._files) > 1:
            raise Exception('Cannot load more than one openml dataset at once')

        df = fetch_openml(name=self._files[0], as_frame=True)

        x = pd.DataFrame(data = df['data'], columns = df['feature_names'])
        y = pd.DataFrame(data = df['target'])

        return pd.concat(list([x, y]), axis=1)
