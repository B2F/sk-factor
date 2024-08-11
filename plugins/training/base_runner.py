from abc import abstractmethod
import pandas as pd

class BaseRunner():

    _config: dict
    _x: pd.DataFrame
    _y: pd.DataFrame
    _labels: pd.DataFrame
    _identifier: str

    def __init__(self, config, x, y, labels, identifier = None):
        self._config = config
        self._x = x
        self._y = y
        self._labels = labels
        self._identifier = identifier

    @abstractmethod
    def run(self, pipeline, cv):
        self._pipeline = pipeline
        self._cv = cv
        # Override directly to implement training without plot, or use the BasePlot class.
        pass
