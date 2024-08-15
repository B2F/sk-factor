from abc import ABC, abstractmethod

class BaseSelector(ABC):

    _config = dict

    def __init__(self, config):

        self._config = config

    @abstractmethod
    def estimator(self) -> callable:

        pass
