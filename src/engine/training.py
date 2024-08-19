import pandas as pd
from src.engine.pipeline import Pipeline
from src.engine.plugins import Plugins
from src.engine.splits import Split
from src.engine.config import Config

class Training():

    _x: pd.DataFrame
    _y: pd.DataFrame
    _y_labels: list
    _config: Config
    _y_labels: list
    _n_splits: int

    def __init__(
        self,
        config: Config,
        x: pd.DataFrame,
        y: pd.DataFrame,
        labels: list,
    ):

        self._x = x
        self._y = y
        self._config = config
        self._y_labels = labels
        self._n_splits = config.get('training', 'nb_splits')

    def run(self):

        runners = self._config.get('training', 'runners')
        if runners is None:
            return

        # Assemble all estimators (sampling, classifiers ...) in a single pipeline
        pipeline = Pipeline.create(self._config, self._x, self._y)

        for runner in runners:

            args = (self._config, self._x, self._y, self._y_labels, runner)
            runnerObject = Plugins.create('training', runner, *args)
            cvList = Split.getList(self._config, self._x, self._y)
            if cvList:
                for cv in cvList:
                    runnerObject.run(pipeline, cv)
            else:
                runnerObject.run(pipeline, 2)

    def setConfig(self, config):
        """ Used to update config in a GUI.
        """
        self._config = config

    def setX(self, x):

        self._x = x

    def setY(self, y):

        self._y = y
