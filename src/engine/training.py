import pandas as pd
from src.engine.pipeline import Pipeline
from src.engine.plugins import Plugins
from src.engine.splits import Split
from src.engine.config import Config

class Training():

    _x: pd.DataFrame
    _y: pd.DataFrame
    _estimators: list
    _runners: list
    _config: Config
    _y_labels: list
    _n_splits: int
    _split_method: str
    _group: str

    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        estimators: list,
        runners: list,
        config: Config,
        labels: list,
        n_splits: int = None,
        split_method: str = None,
        group: str = None
    ):

        self._x = x
        self._y = y
        self._estimators = estimators
        self._config = config
        self._y_labels = labels
        self._runners = runners
        self.setSplit(n_splits, split_method, group)

    def run(self):

        # Assemble all estimators (sampling, classifiers ...) in a single pipeline
        pipeline = Pipeline.create(self._estimators, self._x, self._y, self._config)

        for runner in self._runners:

            args = (self._config, self._x, self._y, self._y_labels, runner)
            runnerObject = Plugins.create('training', runner, *args)
            cv = Split.cv(self._x, self._y, self._config, self._n_splits, self._split_method, self._group)
            runnerObject.run(pipeline, cv)

    def setRunners(self, runners):

        self._runners = runners

    def setConfig(self, config):

        self._config = config

    def setX(self, x):

        self._x = x

    def setY(self, y):

        self._y = y

    def setSplit(self, n_splits: int = None, method: str = None, group: str = None):

        # Lookup in config if not in args:

        if n_splits is None and self._config.get('training', 'nb_splits'):
            self._n_splits = self._config.get('training', 'nb_splits')
        else:
            self._n_splits = n_splits

        if method is None and self._config.get('training', 'splitting_method'):
            self._split_method = self._config.get('training', 'splitting_method')
        else:
            self._split_method = method

        if group is None and self._config.get('training', 'group_column'):
            self._group = self._config.get('training', 'group_column')
        else:
            self._group = group
