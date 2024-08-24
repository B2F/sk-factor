""" SHAP Beasworms features explainer.
@see https://saferml.com/explainable-ai/a-comprehensive-python-tutorial-for-quickly-diving-into-shap/
"""

import matplotlib.pyplot as plt
import shap
from src.engine.model import Model
from plugins.predictions.base_predictor import BasePredictor

class ShapBeeswarm(BasePredictor):

    def _predict(self, model: Model):

        # We assume the classifier comes as last step in the pipeline.
        explainer = shap.Explainer(model.pipeline[len(model.pipeline.steps)-1], self._x)

        fig, axes = plt.subplots(len(model.labels), 1)

        for iLabel in range(len(model.labels)):

            plt.sca(axes[iLabel])
            shap_values = explainer(self._x)[:,:,iLabel]
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            axes[iLabel].title.set_text(model.labels[iLabel])

        plt.tight_layout()
        plt.subplots_adjust(hspace=1)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
