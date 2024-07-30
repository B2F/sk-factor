import lightgbm as lgb
from pipeline.base_lgbm import BaseLgbm

class LgbmRfClassifier(BaseLgbm):

    def getEstimator(self):

        classifier = lgb.LGBMClassifier(
            boosting_type="rf",
            num_leaves=self._params['num_leaves'],
            colsample_bytree=.5,
            n_estimators=400,
            min_child_weight=5,
            min_child_samples=10,
            subsample=.632,
            subsample_freq=1,
            min_split_gain=0,
            reg_alpha=0,
            reg_lambda=5, # L2 regularization
            n_jobs=3,
            seed=self._params['seed']
        )

        return ('lgbm_rf_classifier', classifier)
