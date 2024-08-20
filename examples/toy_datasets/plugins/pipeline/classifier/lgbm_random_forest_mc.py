import lightgbm as lgb
from plugins.pipeline.classifier.lgbm_random_forest import LgbmRandomForest
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class LgbmRandomForestMc(LgbmRandomForest):

    _type = 'classifier'

    def getEstimator(self):

        # all_labels = self._y.to_numpy().flatten()
        # class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        # class_weight_dict = dict(enumerate(class_weights))

        return super().getEstimator(
            objective="multiclassova",
            random_state=self._config.get('training', 'seed'),
            # class_weight=class_weight_dict,
            boosting_type="rf",
            num_leaves=100,
            colsample_bytree=.5,
            n_estimators=400,
            min_child_weight=5,
            min_child_samples=10,
            subsample=.632,
            subsample_freq=1,
            min_split_gain=0,
            # reg_alpha=10, # Hard L1 regularization
            # reg_lambda=0,
            reg_alpha=0,
            reg_lambda=5, # L2 regularization
            n_jobs=3,
            learning_rate=0.08,
        )
