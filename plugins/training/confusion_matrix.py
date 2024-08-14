from plugins.training.training_plot import TrainingPlot
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class ConfusionMatrix(TrainingPlot):

    def plot(self):

        text = "Classes distribution:\n"
        for index, count in self._y.value_counts().to_dict().items():
            text += str(self._labels[index]) + ': ' + str(count) + '\n'
        # y_values_count = y_train.value_counts().to_list()
        # y_values = np.unique(y_train.values)

        if not type(self._cv) is int or self._cv > 1:
            y_pred = cross_val_predict(self._pipeline, self._x, self._y.values.flatten(), cv = self._cv)
        else:
            y_pred = cross_val_predict(self._pipeline, self._x, self._y.values.flatten())

        # @todo add a normalize='true optionnal param to better understand classes imbalance.
        conf_matrix = confusion_matrix(self._y, y_pred)
        cm_df = pd.DataFrame(conf_matrix,
                        index = self._labels,
                        columns = self._labels)

        fig, ax = plt.subplots(2, figsize=(10, 6))
        ax[1].set_axis_off()
        ax[1].text(0, 1, text, va='top', ha='right')
        ax[0].set_title('Confusion matrix', pad=20)
        ax[0].xaxis.set_ticks_position('top')
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlOrBr', ax = ax[0])
        ax[1].margins(10, tight=False)

        return 'confusion_matrix'
