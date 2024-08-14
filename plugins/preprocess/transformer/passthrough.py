from plugins.preprocess.base_transformer import BaseTransformer

class Passthrough(BaseTransformer):

    _name = 'passthrough'

    def estimator(self):
        # Default method, used by the passthrough transformer:
        return super().estimator()
