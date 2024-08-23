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
    _labels: list

    def __init__(self, config, files, labels = None, models = []):
        """ Model is specified in config by default,
        The trained model is used otherwise.
        """

        self._labels = labels
        self._config = config
        self._models = []

        loader = config.get('predictions', 'loader')
        df = Plugins.create('loader', loader, config, files).load()

        if config.get('predictions', 'preprocess'):

            df = Transformers.apply(config, df)

            print('\nBefore predictions preprocessing:')
            print(df.shape)
            df = Preprocessors.apply(config, df)

        self._x = df

        modelsFilePath = self._config.get('predictions', 'models')
        if modelsFilePath is not None:
            for modelFilePath in modelsFilePath:
                model = Model(config, df, [], joblib.load(modelFilePath), modelsFilePath)
                self._models.append(model)
        elif models:
            self._models = models
        else:
            raise Exception('Predictions models are missing from both config and training, you need at least one model with predictions enabled.')

    def run(self):

        objective = self._config.get('predictions', 'objective')
        predictor = Plugins.create('predictions', objective, self._config, self._x, self._labels)

        for model in self._models:
            predictor.predict(model)
