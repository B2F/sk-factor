from sklearn.linear_model import RidgeCV as RidgeBase
from plugins.estimators.base_estimator import BaseEstimator
import numpy as np

class RidgeCv(BaseEstimator):

    _type = 'regressor'

    def getEstimator(self):

        alphas = np.logspace(-6, 6, 13)

        # Create an instance of RidgeCV with the defined alphas
        return ('ridge', RidgeBase(alphas=alphas))
