from plugins.preprocess.transformer.simple_imputer import SimpleImputer
from pathlib import Path

class HousingSimpleImputer(SimpleImputer):

    _name = Path(__file__).stem

    def estimator(self):

        return super().estimator(add_indicator = True)
