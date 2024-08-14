import pandas as pd
from abc import ABC, abstractmethod

class BaseLoader(ABC):

    _config: dict
    _arguments: list

    def __init__(self, config: dict, arguments: list = []):

        self._arguments = arguments
        self._config = config

    @abstractmethod
    def load() -> pd.DataFrame:

        pass
