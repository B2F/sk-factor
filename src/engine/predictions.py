import joblib
import pandas as pd
from src.engine.preprocessors import Preprocessors
from src.engine.transfomers import Transformers
from src.engine.plugins import Plugins

class Predictions():

    _x: pd.DataFrame
    _plugin: Plugins
    _config: dict
    _model: object
    _labels: list

    def __init__(self, config, files, labels = None, model = None):
        """ Model is specified in config by default,
        The trained model is used otherwise.
        """

        self._labels = labels
        self._config = config
        loader = config.get('predictions', 'loader')
        df = Plugins.create('loader', loader, config, files).load()

        if config.get('predictions', 'preprocess'):
          x, y_train, labels = Preprocessors.apply(config, df)
          x = Transformers.apply(config, x)
        else:
            x = df

        self._x = x

        modelFilePath = self._config.get('predictions', 'model')
        if modelFilePath:
            self._model = joblib.load(modelFilePath)
        elif model is not None:
            self._model = model
        else:
            raise Exception('Predictions model is missing from both config and training, you need at least one model with predictions enabled.')

    def run(self):

        objective = self._config.get('predictions', 'objective')
        predictor = Plugins.create('predictions', objective, self._config, self._x, self._labels)
        predictor.predict(self._model)

