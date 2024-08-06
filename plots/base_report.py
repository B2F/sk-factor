import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import time
import os

class Report(ABC):

    _config: dict
    _x: pd.DataFrame
    _y: pd.DataFrame
    _labels: pd.DataFrame
    _showPlot: bool
    _saveImage: bool
    _saveTimestamp: str
    _imagesExtension: str
    _imagesDirectory: str
    __identifier: str

    def __init__(self, config, x, y, labels, identifier):
        self._config = config
        self._x = x
        self._y = y
        self._labels = labels
        self._showPlot = eval(config['eda']['show_plots'])
        self._saveImage = eval(config['eda']['save_images'])
        self._saveTimestamp = eval(config['eda']['save_timestamp'])
        self._imagesDirectory = config['eda']['images_directory']
        self._imagesExtension = config['eda']['images_extension']
        self.__identifier = identifier

    def getImageFilepath(self, filename):
        directory = f'{self._imagesDirectory}/{self.__identifier}'
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if self._saveTimestamp:
            filename += '_' + str(time.time())
        filename = f"{filename}.{self._imagesExtension}"
        return f'{directory}/{filename}'

    @abstractmethod
    def run(self, plotId):
        if self._showPlot:
            plt.tight_layout()
            plt.show()
        if self._saveImage:
            plt.savefig(self.getImageFilepath(plotId))
            print(self.__class__.__name__ + f': {self.getImageFilepath(plotId)} written to disk')
