import joblib
import pandas as pd
from src.engine.preprocessors import Preprocessors
from src.engine.transfomers import Transformers
from src.engine.plugins import Plugins
from src.engine.model import Model

class Predictions():

    _x: pd.DataFrame
    _plugin: Plugins
    _config: dict
    _models: list

    def __init__(self, config, files, models = []):
        """ Model is specified in config by default,
        The trained model is used otherwise.
        """

        self._config = config
        self._models = []

        loader = config.get('predictions', 'loader')
        df = Plugins.create('loader', loader, config, files).load()

        # We remove y in case it is provided with 'drop_rows_to_predict_file':
        if self._config.get('preprocess', 'label') in df.columns:
            self._x, y, labels = Preprocessors.encodeLabel(config, df)

        # You usually won't use preprocessing on prediction data.
        if config.eq('predictions', 'preprocess', True):

            self._x = Transformers.apply(config, self._x)

            print('\nBefore predictions preprocessing:')
            print(self._x.shape)
            self._x = Preprocessors.apply(config, self._x)

        modelsFilePath = self._config.get('predictions', 'models')
        if modelsFilePath is not None:
            for modelFilePath in modelsFilePath:
                model = joblib.load(modelFilePath)
                self._models.append(model)
        elif models:
            self._models = models
        else:
            raise Exception('Predictions models are missing from both config and training, you need at least one model with predictions enabled.')

    def run(self):

        objective = self._config.get('predictions', 'objective')
        predictor = Plugins.create('predictions', objective, self._config, self._x)

        for model in self._models:
            predictor.predict(model)
