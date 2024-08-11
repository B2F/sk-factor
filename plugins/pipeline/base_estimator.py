from abc import ABC, abstractmethod

class BaseEstimator(ABC):

    TYPES = ('classifier', 'regressor', 'transformer', 'sampler')

    _type = str

    def __init__(self, config, x, y):

        self._config = config
        self._x = x
        self._y = y

    @abstractmethod
    def getEstimator(self) -> tuple:
        '''
        Returns a pipeline tuple such as ('classifier', RandomForestClassifier())
        '''
        pass
