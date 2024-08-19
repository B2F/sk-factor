"""Records a model file from training data."""

import joblib
import os
import pandas as pd
import time
from sklearn.pipeline import Pipeline

class Model:

    _config: dict
    _saveModel: bool
    _modelTimestamp: bool
    _modelsDirectory: bool
    _x: pd.DataFrame
    _y: pd.DataFrame
    _pipeline: Pipeline

    def __init__(self, config, x, y, pipeline):

        self._config = config
        self._saveModel = config.get('training', 'save_model')
        self._modelTimestamp = config.get('training', 'model_timestamp')
        self._modelsDirectory = config.get('training', 'models_directory')
        self._x = x
        self._y = y
        self._pipeline = pipeline

    def save(self):

        if not self._saveModel:
            return

        self._pipeline.fit(self._x, self._y)
        # The filename is retrieved from config file, in sk_factor.py
        filename = self._config.get('dataset', 'filename')
        if self._modelTimestamp:
            filename = filename + time.time()

        modelFilePath = self._modelsDirectory + f'/{filename}.pkl'
        if not os.path.isdir(self._modelsDirectory):
            os.makedirs(self._modelsDirectory)

        joblib.dump(self._pipeline, modelFilePath)

        print ('Model file written in ' + modelFilePath)
