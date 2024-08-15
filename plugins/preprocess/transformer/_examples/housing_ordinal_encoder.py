from plugins.preprocess.transformer.ordinal_encoder import OrdinalEncoder
from pathlib import Path

class HousingOrdinalEncoder(OrdinalEncoder):

    _name = Path(__file__).stem

    def estimator(self):

        return super().estimator(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
