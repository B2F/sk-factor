import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from matplotlib import rcParams
from pathlib import Path

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

    def __init__(self, config, x, y, labels, identifier = None):

        if identifier == None:
            identifier = Path(__file__).stem

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
        # figure size in inches
        if type(eval(config['eda'].get('figsize'))) is tuple:
            rcParams['figure.figsize'] = config['eda']['figsize']

    def getImageFilepath(self, filename):
        imagePath = self.__identifier.replace('/', '-')
        directory = f'{self._imagesDirectory}/{imagePath}'
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if self._saveTimestamp:
            filename += '_' + str(time.time())
        filename = f"{filename}.{self._imagesExtension}"
        return f'{directory}/{filename}'

    @abstractmethod
    def plot(self) -> str:
        pass

    def run(self):
        plotId = self.plot()
        dpi = 100
        if self._config['eda'].get('dpi'):
            dpi=self._config['eda']['dpi']
        if self._showPlot:
            plt.tight_layout()
            plt.show()
        if self._saveImage:
            plt.savefig(self.getImageFilepath(plotId), dpi=dpi)
            print(os. getcwd()  + '/' + self.getImageFilepath(plotId) + " written to disk")
        plt.clf()
