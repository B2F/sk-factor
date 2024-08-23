from sklearn import svm
from plugins.estimators.base_estimator import BaseEstimator

class LinearSvc(BaseEstimator):

    _type = 'classifier'

    def getEstimator(self):
        return ('linear_svc', svm.SVC(
            kernel='linear',
            probability=True,
            random_state=self._config.get('training', 'seed')
        ))
