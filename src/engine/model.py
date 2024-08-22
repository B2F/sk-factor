"""Records a model file from training data."""

import joblib
import os
import pandas as pd
import re
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
    _id: str

    def __init__(self, config, x, y, pipeline, id):

        self._config = config
        self._saveModel = config.get('training', 'save_model')
        self._modelTimestamp = config.get('training', 'model_timestamp')
        self._modelsDirectory = config.get('training', 'models_directory')
        self._x = x
        self._y = y
        self._pipeline = pipeline
        self._id = id

    @property
    def id(self):
        return self._id

    @property
    def pipeline(self):
        return self._pipeline

    def fit(self):
        self._pipeline.fit(self._x, self._y.to_numpy().flatten())

    def save(self):

        # The default model filename is retrieved from config file, in sk_factor.py
        filename = self._config.get('dataset', 'filename') + '-' + self._id

        if self._modelTimestamp:
            filename = filename + '_' + str(time.time())

        modelFilePath = self._modelsDirectory + f'/{filename}.pkl'
        modelFileDir = re.match(r'^(.*\/)+', self._modelsDirectory + f'/{filename}').group(0)
        if not os.path.isdir(modelFileDir):
            os.makedirs(modelFileDir)

        joblib.dump(self._pipeline, modelFilePath)
        print ('Model file written in ' + modelFilePath)
