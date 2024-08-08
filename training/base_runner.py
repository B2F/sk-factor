from abc import abstractmethod

class BaseRunner():

    @abstractmethod
    def run(self, pipeline, cv):
        self._pipeline = pipeline
        self._cv = cv
        # Override directly to implement training without plot, or use the BasePlot class.
        pass
