import pandas as pd
from src.engine.plugins import Plugins
from src.engine.config import Config
import importlib
from src.engine.config import Config
from plugins.estimators.base_estimator import BaseEstimator

class BasePipeline:

    _config: dict
    _steps: list
    _x: pd.DataFrame
    _y: pd.DataFrame

    def __init__(self, config: Config, x, y):
        self._config = config
        self._steps = []
        pipelineModule = config.get('training', 'pipeline')
        pipeline = importlib.import_module(pipelineModule)
        self._pipeline = getattr(pipeline, 'Pipeline')
        self._x = x
        self._y = y

    def getPipeline(self):
        return self._pipeline(
            steps = self._steps
        )

    def addStep(self, estimator: BaseEstimator):
        estimator = Plugins.create('estimators', estimator, self._config, self._x, self._y)
        self._steps.append(estimator.getEstimator())

class Pipeline():
    """ Pipeline Factory.
    """

    @staticmethod
    def create(config: Config, x: pd.DataFrame, y: pd.DataFrame) -> list:
        """ Returns a list of tuples (ID, pipeline) from given config + dataframes.
        """

        pipelines = []

        # Assemble one pipeline per target estimator (classifier, regressor, grid search ...)
        for estimator in config.get('training', 'estimators'):

            factoredPipeline = BasePipeline(config, x, y)

            # Each pipeline can have a stack of samplers and transformers.
            samplers = config.get('training', 'samplers') or []
            for sampler in samplers:
                factoredPipeline.addStep(sampler)
            transformers = config.get('training', 'transformers') or []
            for transformer in transformers:
                factoredPipeline.addStep(transformer)

            # Add the target estimator as last step:
            factoredPipeline.addStep(estimator)
            pipelines.append((estimator, factoredPipeline.getPipeline()))

        return pipelines
