import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import time
import os
from matplotlib import rcParams
from pathlib import Path
from src.engine.config import Config

class Report(ABC):

    _config: Config
    _x: pd.DataFrame
    _y: pd.DataFrame
    _labels: pd.DataFrame
    _showPlot: bool
    _saveImage: bool
    _saveTimestamp: str
    _imagesExtension: str
    _imagesDirectory: str
    _identifier: str

    def __init__(self, config: Config, x, y, labels, identifier = None):

        if identifier == None:
            identifier = Path(__file__).stem

        self._config = config

        if config.get('eda', 'features'):
            self._x = x[config.get('eda', 'features')]
        else:
            self._x = x

        self._y = y

        self._labels = labels
        self._showPlot = config.get('eda', 'show_plots')
        self._saveImage = config.get('eda', 'save_images')
        self._saveTimestamp = config.get('eda', 'save_timestamp')
        self._imagesDirectory = config.get('eda', 'images_directory')
        self._imagesExtension = config.get('eda', 'images_extension')
        self._identifier = identifier

        # figure size in inches
        if type(config.get('eda', 'figsize')) is list:
            rcParams['figure.figsize'] = (config.get('eda', 'figsize')[0], config.get('eda', 'figsize')[1])

    def getImageFilepath(self, filename):
        imagePath = self._identifier.replace('/', '-')
        directory = f'{self._imagesDirectory}/{imagePath}'
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if self._saveTimestamp:
            # @todo: convert timestamp to date in the predictions output (do the same for predictions and model file).
            filename += '_' + str(time.time())
        filename = f"{filename}.{self._imagesExtension}"
        return f'{directory}/{filename}'

    @abstractmethod
    def plot(self) -> str:
        pass

    def run(self):

        plotId = self.plot()
        dpi = 100

        if self._config.get('eda', 'dpi'):
            dpi=self._config.get('eda', 'dpi')

        if self._showPlot:
            plt.show()
        if self._saveImage:
            plt.savefig(self.getImageFilepath(plotId), dpi=dpi)
            print(os. getcwd()  + '/' + self.getImageFilepath(plotId) + " written to disk")

        plt.close()
