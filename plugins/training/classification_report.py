from plugins.training.base_runner import BaseRunner
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
import numpy as np

class ClassificationReport(BaseRunner):

    def run(self, pipeline, cv):

        y_pred = cross_val_predict(pipeline, self._x, self._y.values.flatten(), cv=cv)

        # Convert labels to strings if they are numeric, similar to confusion_matrix.py
        target_names = self._labels
        if self._labels is not None:
            # Convert to list of strings to handle numeric labels properly
            if hasattr(self._labels, '__iter__') and not isinstance(self._labels, str):
                try:
                    target_names = [str(label) for label in self._labels]
                except:
                    target_names = None
            else:
                target_names = [str(self._labels)]

        print(classification_report(self._y, y_pred, target_names=target_names))
