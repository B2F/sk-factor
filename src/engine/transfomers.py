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
    def apply(transformers: dict, many_to_one: dict, config: Config, x: pd.DataFrame):

        verboseFeatureNamesOut = config.eq('preprocess', 'verbose_feature_names_out', True)
        encoders = []

        for module, features in transformers.items():

            transformer = Plugins.create('preprocess.transformer', module, config.getConfig(), x)
            # Preprocess for columns selected in the .ini config.
            features = list(x.columns) if features == [] else features
            encoders.append((module, transformer.pipeline(), features))

        for feature, transformers in many_to_one.items():

            transformer = Plugins.create(
                'preprocess.transformer',
                'many_to_one',
                config.getConfig(),
                x,
                transformers,
                feature
            )

            encoders.append((f'many_to_one_{feature}', transformer.pipeline(), [feature]))

        preprocessor = ColumnTransformer(
            transformers=encoders,
            verbose_feature_names_out=verboseFeatureNamesOut
        )

        x = preprocessor.set_output(transform="pandas").fit_transform(x)

        if verboseFeatureNamesOut and 'passthrough__group' in x.columns:
            x.drop('passthrough__group', axis=1, inplace=True)

        return x
