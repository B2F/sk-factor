import sklearn.datasets as toy_datasets
import pandas as pd
from plugins.loader.base_loader import BaseLoader

class ToyDataset(BaseLoader):

    def _load(self):

        if len(self._arguments) > 1:
            raise Exception('Cannot load more than one toy dataset at once')

        dataset = getattr(toy_datasets, f'load_{self._arguments[0]}')()

        x = pd.DataFrame(data = dataset['data'], columns = dataset['feature_names'])
        y = pd.DataFrame(data = dataset['target'], columns = ['target'])

        if dataset.get('target_names') is not None:
            labelValues = [list(dataset['target_names']).index(x) for x in list(dataset['target_names'])]
            y.replace(labelValues, dataset['target_names'], inplace=True)

        return pd.concat(list([x, y]), axis=1)
