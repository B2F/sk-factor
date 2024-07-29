import pandas as pd
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):

    @abstractmethod
    def preprocess(self, df) -> pd.DataFrame:
        # Preprocess the whole dataset at once here.
        return self._df
