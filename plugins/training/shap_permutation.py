""" SHAP features permutation (importance).
@see https://saferml.com/explainable-ai/a-comprehensive-python-tutorial-for-quickly-diving-into-shap/
"""

import shap
import matplotlib.pyplot as plt
from plugins.training.training_plot import TrainingPlot

class ShapPermutation(TrainingPlot):

    def plot(self):

        # We assume the classifier comes as last step in the pipeline.
        self._pipeline.fit(self._x, self._y.to_numpy().flatten())
        explainer = shap.explainers.Permutation(self._pipeline.predict_proba, self._x)
        shap_values = explainer(self._x)

        fig, ax = plt.subplots(nrows=len(self._labels), ncols=1)

        for iLabel in range(len(self._labels)):

            shap.plots.bar(shap_values[:,:,iLabel], ax=ax[iLabel], show=False)
            ax[iLabel].title.set_text(self._labels[iLabel])

        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
