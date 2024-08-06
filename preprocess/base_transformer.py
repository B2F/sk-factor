from abc import abstractmethod
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

class BaseTransformer():

    _action = str
    _config = dict
    _df = pd.DataFrame

    def __init__(self, config, df):
        self._action = Path(__file__).stem
        self._config = config
        self._df = df

    @abstractmethod
    def transform(self) -> pd.DataFrame:
        # Dummy method used by the passthrough transformer:
        return FunctionTransformer()

    def pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (self._action, self.transform())
            ]
        )
