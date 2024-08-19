import pandas as pd
from src.engine.plugins import Plugins
from src.engine.config import Config
import importlib
from src.engine.config import Config

class BasePipeline:

      _steps: list

      def __init__(self, config: Config):
          self._steps = []
          pipelineModule = config.get('training', 'pipeline')
          pipeline = importlib.import_module(pipelineModule)
          self._pipeline = getattr(pipeline, 'Pipeline')

      def getPipeline(self):
          return self._pipeline(
              steps = self._steps
          )

      def addStep(self, stepTuple):
          self._steps.append(stepTuple)

class Pipeline():

    @staticmethod
    def create(config: Config, x: pd.DataFrame, y: pd.DataFrame):

        # Assemble all estimators (sampling, classifiers ...) in a single pipeline
        factoredPipeline = BasePipeline(config)
        for estimator in config.get('training', 'estimators'):
            # Put GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV etc in the plugins pipeline directory.
            # Same for automated feature selection ? (RFECV, SelectFromModel)
            estimator = Plugins.create('pipeline', estimator, config, x, y)
            factoredPipeline.addStep(estimator.getEstimator())

        return factoredPipeline.getPipeline()
