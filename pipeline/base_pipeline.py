import importlib

class BasePipeline:

      _steps: list

      def __init__(self, config):
          self._steps = []
          pipelineModule = config['training']['pipeline']
          pipeline = importlib.import_module(pipelineModule)
          self._pipeline = getattr(pipeline, 'Pipeline')

      def getPipeline(self):
          return self._pipeline(
              steps = self._steps
          )

      def addStep(self, stepTuple):
          self._steps.append(stepTuple)
