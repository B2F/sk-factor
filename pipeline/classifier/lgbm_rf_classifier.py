import lightgbm as lgb
from pipeline.base_lgbm import BaseLgbm

class LgbmRfClassifier(BaseLgbm):

    def getEstimator(self):

        self._params['boosting_type'] = 'rf'
        classifier = lgb.LGBMClassifier(**self._params)
        return ('lgbm_rf_classifier', classifier)
