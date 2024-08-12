from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.engine.config import Config
from sklearn.compose import ColumnTransformer
from src.engine.plugins import Plugins

class Transformers():

    @staticmethod
    def labelEncode(y: pd.DataFrame, labelName: str):

        le = LabelEncoder()
        y = pd.DataFrame(list(le.fit_transform(y.values.flatten())), columns=[labelName])
        return (y, le.classes_)

    @staticmethod
    def apply(transformers: dict, x: pd.DataFrame, config: Config):

        verboseFeatureNamesOut = config.eq('preprocess', 'verbose_feature_names_out', True)
        encoders = []

        for action, features in transformers.items():

            transformer = Plugins.create('preprocess', 'transformer/' + action, config.getConfig(), x)
            # Preprocess for columns selected in the .ini config.
            features = list(x.columns) if features == [] else features
            encoders.append((action, transformer.pipeline(), features))

        preprocessor = ColumnTransformer(
            transformers=encoders,
            verbose_feature_names_out=verboseFeatureNamesOut
        )

        x = preprocessor.set_output(transform="pandas").fit_transform(x)

        if verboseFeatureNamesOut and 'passthrough__group' in x.columns:
            x.drop('passthrough__group', axis=1, inplace=True)

        return x
