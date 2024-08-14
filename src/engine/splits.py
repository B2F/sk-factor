import pandas as pd
from src.engine.plugins import Plugins

class Split():

    @staticmethod
    def cv(
        config: dict,
        x: pd.DataFrame,
        y: pd.DataFrame,
        n_splits: int,
    ):

        method = config.get('training', 'splitting_method')
        group = config.get('training', 'group_column')

        if group is not None and group in x.columns:
            group = x[group]
        if n_splits > 1 and method is not None:
            iteratorObject = Plugins.create('split', method, config, x, y, n_splits, group)
            cv = iteratorObject.split()
        else:
            cv = n_splits

        return cv
