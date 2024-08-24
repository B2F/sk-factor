from plugins.training.training_plot import TrainingPlot
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.model_selection import cross_val_predict

class PrecisionRecall(TrainingPlot):

    def plot(self):

        if not type(self._cv) is int or self._cv > 1:
            y_pred = cross_val_predict(self._pipeline, self._x, self._y, cv = self._cv)
        else:
            y_pred = cross_val_predict(self._pipeline, self._x, self._y)

        display = PrecisionRecallDisplay.from_predictions(
            self._y, y_pred, plot_chance_level=True
        )
        display.ax_.set_title("2-class Precision-Recall curve")
        precision, recall, thresholds = precision_recall_curve(self._y, y_pred)
        for i, threshold in enumerate(thresholds):
            display.ax_.annotate(f'{threshold:.2f}', (recall[i], precision[i]))
