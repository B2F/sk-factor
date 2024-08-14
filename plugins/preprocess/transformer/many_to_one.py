from plugins.preprocess.base_transformer import BaseTransformer
from sklearn.preprocessing import FunctionTransformer
from src.engine.plugins import Plugins

class ManyToOne(BaseTransformer):

    _name = 'many_to_one'
    _transformers: list
    _feature: str

    def __init__(self, config, df, transformers, feature):
      super().__init__(config, df)
      self._transformers = transformers
      self._feature = feature

    def getSteps(self):

        transformers = []
        for module in self._transformers:
            transformer = Plugins.create('preprocess.transformer', module, self._config, self._df)
            transformers.append(
                ( self._feature + '_' + module, transformer.estimator() )
            )

        return transformers
