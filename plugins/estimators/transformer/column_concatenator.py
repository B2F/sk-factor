from sktime.transformations.panel.compose import ColumnConcatenator as BaseColumnConcatenator
from plugins.estimators.base_estimator import BaseEstimator

class ColumnConcatenator(BaseEstimator):

    _type = 'transformer'

    def getEstimator(self) -> tuple:

        cc = BaseColumnConcatenator()
        return ('column_concatenator', cc)
