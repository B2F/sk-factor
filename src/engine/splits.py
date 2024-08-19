import pandas as pd
from src.engine.plugins import Plugins

class Split():

    @staticmethod
    def getList(
        config: dict,
        x: pd.DataFrame,
        y: pd.DataFrame,
    ) -> list:

        list = []
        splittingMethods = config.get('training', 'splitting_method')
        if splittingMethods is None:
            return

        for plugin, n_splits in splittingMethods.items():
            if n_splits > 1:
                iteratorObject = Plugins.create('split', plugin, config, x, y, n_splits)
                list.append(iteratorObject.split())

        return list
