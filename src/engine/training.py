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
        n_splits: int,
    ):

        self._x = x
        self._y = y
        self._config = config
        self._y_labels = labels
        self._n_splits = n_splits

    def run(self):

        runners = self._config.get('training', 'runners')
        if runners is None:
            return

        # Assemble all estimators (sampling, classifiers ...) in a single pipeline
        pipeline = Pipeline.create(self._config, self._x, self._y)

        for runner in runners:

            args = (self._config, self._x, self._y, self._y_labels, runner)
            runnerObject = Plugins.create('training', runner, *args)
            cv = Split.cv(self._config, self._x, self._y, self._n_splits)
            runnerObject.run(pipeline, cv)

    def setConfig(self, config):

        self._config = config

    def setX(self, x):

        self._x = x

    def setY(self, y):

        self._y = y
