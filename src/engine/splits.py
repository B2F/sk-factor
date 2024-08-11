import pandas as pd
from src.engine.plugins import Plugins

class Split():

    @staticmethod
    def cv(
        x: pd.DataFrame,
        y: pd.DataFrame,
        config: dict,
        n_splits: int,
        method: str = None,
        group: str = None
    ):

        if group is not None and group in x.columns:
            group = x[group]
        if n_splits > 1 and method is not None:
            iteratorObject = Plugins.create('split', method, config, x, y, n_splits, group)
            cv = iteratorObject.split()
        else:
            cv = n_splits

        return cv
