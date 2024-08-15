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
    def apply(config: Config, x: pd.DataFrame):

        transformers = config.get('preprocess', 'transformers')
        many_to_one = config.get('preprocess', 'transformers_many_to_one')

        verboseFeatureNamesOut = config.eq('preprocess', 'verbose_feature_names_out', True)
        encoders = []

        for module, features in transformers.items():

            transformer = Plugins.create('preprocess.transformer', module, config.getConfig(), x)
            if type(features) is list:
                # Preprocess for columns selected in the .ini config.
                features = list(x.columns) if features == [] else features
            elif type(features) is str:
                features = Plugins.create('preprocess.selector', features, config).estimator()
            else:
                raise Exception(module + ' value type is unsupported, only list and str are valid transformers types.')
            encoders.append((module, transformer.pipeline(), features))

        many_to_one_transformers = [] if not many_to_one else many_to_one.items()
        for feature, transformers in many_to_one_transformers:

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
