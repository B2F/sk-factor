import lightgbm as lgb
from plugins.pipeline.base_lgbm import BaseLgbm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class LgbmRandomForest(BaseLgbm):

    _type = 'classifier'

    def getEstimator(self):

        classifier = lgb.LGBMClassifier(
            verbosity=-1,
            objective="binary",
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
            seed=self._params['seed'],
            learning_rate=0.08,
        )

        return ('lgbm_random_forest', classifier)
