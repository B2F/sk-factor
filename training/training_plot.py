from plots.base_report import Report
from training.base_runner import BaseRunner
from abc import abstractmethod

class TrainingPlot(Report, BaseRunner):

    def __init__(self, config, x, y, labels, identifier = None):

        Report.__init__(self, config, x, y, labels, identifier)
        if self._config['training'].get('show_plots'):
            self._showPlot = self._config['training']['show_plots']
        if self._config['training'].get('save_images'):
            self._saveImage = self._config['training']['save_images']

    @abstractmethod
    def plot(self):
        # @see training/confusion_matrix
        pass

    def run(self, pipeline, cv):
        self._pipeline = pipeline
        self._cv = cv
        Report.run(self)
