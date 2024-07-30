import lightgbm as lgb
from pipeline.base_lgbm import BaseLgbm

class LgbmRfRegressor(BaseLgbm):

    def getEstimator(self):

        self._params['boosting_type'] = 'rf'
        # classifier = lgb.LGBMRegressor(**self._params)
        classifier = lgb.LGBMRegressor()
        return ('lgbm_rf_regressor', classifier)
