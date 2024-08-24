from imblearn.over_sampling import SMOTE
from plugins.estimators.base_estimator import BaseEstimator

class Smote(BaseEstimator):

    _type = 'sampler'

    def getEstimator(self) -> tuple:

        smote = SMOTE()
        return ('smote', smote)
