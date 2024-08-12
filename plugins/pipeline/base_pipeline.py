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
