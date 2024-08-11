from plugins.pipeline.base_lgbm import BaseLgbm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class BeansRandomForest(BaseLgbm):

    _type = 'classifier'

    def getEstimator(self):

        all_labels = self._y.to_numpy().flatten()
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        count = 0
        class_weight_dict = {}
        for weight in class_weights:
            class_weight_dict[count] = weight
            count += 1

        return ('beans_random_forest', RandomForestClassifier(class_weight=class_weight_dict))
