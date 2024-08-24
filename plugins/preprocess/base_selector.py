from abc import ABC, abstractmethod

class BaseSelector(ABC):
    """ Feature selection plugins.
    https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
    """

    _config = dict

    def __init__(self, config):

        self._config = config

    @abstractmethod
    def estimator(self) -> callable:

        pass
