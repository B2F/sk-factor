from abc import ABC, abstractmethod

class BaseEstimator(ABC):

    @abstractmethod
    def getEstimator(self) -> tuple:
        '''
        Returns a pipeline tuple such as ('classifier', RandomForestClassifier())
        '''
        pass
