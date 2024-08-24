from plugins.estimators.classifier.lgbm_classifier import LgbmClassifier
import numpy as np

class LgbmRandomForestMc(LgbmClassifier):

    _type = 'classifier'

    def getEstimator(self):

        return super().getEstimator(
            objective="multiclassova",
            class_weight=self._classWeights,
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
