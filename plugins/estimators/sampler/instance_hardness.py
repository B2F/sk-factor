from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import InstanceHardnessThreshold
from plugins.estimators.base_estimator import BaseEstimator

class InstanceHardness(BaseEstimator):

    _type = 'sampler'

    def getEstimator(self) -> tuple:

        iht = InstanceHardnessThreshold(random_state=0, estimator=LogisticRegression())
        return ('instance_hardness', iht)
