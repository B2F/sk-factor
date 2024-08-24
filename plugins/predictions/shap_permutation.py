""" SHAP Beasworms features explainer.
@see https://saferml.com/explainable-ai/a-comprehensive-python-tutorial-for-quickly-diving-into-shap/
"""

import shap
import matplotlib.pyplot as plt
from src.engine.model import Model
from plugins.predictions.base_predictor import BasePredictor

class ShapPermutation(BasePredictor):

    def _predict(self, model: Model):

        # We assume the classifier comes as last step in the pipeline.
        explainer = shap.explainers.Permutation(model.pipeline.predict_proba, self._x)
        shap_values = explainer(self._x)

        fig, ax = plt.subplots(nrows=len(model.labels), ncols=1)

        for iLabel in range(len(model.labels)):

            shap.plots.bar(shap_values[:,:,iLabel], ax=ax[iLabel], show=False)
            ax[iLabel].title.set_text(model.labels[iLabel])

        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
