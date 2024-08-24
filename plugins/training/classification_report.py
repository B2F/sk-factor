from plugins.training.base_runner import BaseRunner
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

class ClassificationReport(BaseRunner):

    def run(self, pipeline, cv):

        y_pred = cross_val_predict(pipeline, self._x, self._y.values.flatten(), cv=cv)
        print(classification_report(self._y, y_pred, target_names=self._labels))
