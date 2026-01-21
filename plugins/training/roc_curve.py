from plugins.training.training_plot import TrainingPlot
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_predict
from src.utils.data_validator import validate_numeric_columns

class RocCurve(TrainingPlot):

    def plot(self):
        # Validate that all columns in X can be converted to numeric
        validate_numeric_columns(self._x, "ROCCurve training data (X)")
        
        if not type(self._cv) is int or self._cv > 1:
            y_pred = cross_val_predict(self._pipeline, self._x, self._y, cv = self._cv)
        else:
            y_pred = cross_val_predict(self._pipeline, self._x, self._y)

        display = RocCurveDisplay.from_predictions(
            self._y, y_pred, plot_chance_level=True
        )
        display.ax_.set_title("2-class ROC curve")
        
        return 'roc_curve'
