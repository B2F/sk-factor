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

    def __init__(self, config, files, model = None):
        """ Model is specified in config by default,
        The trained model is used otherwise.
        """

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
        predictor = Plugins.create('predictions', objective, self._config, self._x)
        predictions = predictor.predict(self._model)

        print(predictions)

        # if self._config.get('predictions', 'save_predictions'):

        """
predictions_directory = 'output/predictions'
save_predictions = false
predictions_timestamp = true
# Keep X data features in the final predictions columns.
keep_data = true"""
