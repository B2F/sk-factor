""" SHAP Beeswarms features explainer.
@see https://saferml.com/explainable-ai/a-comprehensive-python-tutorial-for-quickly-diving-into-shap/
"""

import matplotlib.pyplot as plt
import shap
from plugins.training.training_plot import TrainingPlot

class ShapBeeswarm(TrainingPlot):

    def plot(self):

        # We assume the classifier comes as last step in the pipeline.
        self._pipeline.fit(self._x, self._y.to_numpy().flatten())
        explainer = shap.Explainer(self._pipeline[len(self._pipeline.steps)-1], self._x)

        fig, axes = plt.subplots(len(self._labels), 1)

        for iLabel in range(len(self._labels)):

            plt.sca(axes[iLabel])
            shap_values = explainer(self._x)[:,:,iLabel]
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            axes[iLabel].title.set_text(self._labels[iLabel])

        plt.tight_layout()
        plt.subplots_adjust(hspace=1)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
