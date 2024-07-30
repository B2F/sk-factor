import pandas as pd
import argparse
import configparser
import importlib
from sklearn.compose import ColumnTransformer
from pipeline.base_pipeline import BasePipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help = "Use a config file (cli args take precedence, similar keys)", default="ml_factor.ini")

# -t and -p arguments can be cumulated
parser.add_argument("-t", "--train_files", help = "Train with given file(s)", required = False, nargs = "*")
parser.add_argument("-p", "--test_files", help = "Predict with given file(s)", required = False, nargs = "*")
parser.add_argument("-m", "--model_file", help = "Model file(s) used for predictions", required = False, nargs = "*")

argument = parser.parse_args()

config = configparser.ConfigParser()
config.read(argument.config)

if eval(config['debug']['enabled']) == True:
    debugpy = importlib.import_module("debugpy")
    debugpy.listen((config['debug']['host'], eval(config['debug']['port'])))
    debugpy.breakpoint()

# Modules loading
def getClassFromConfig(package, config):
    # Only one directory sub level is supported atm.
    if config.find('/') != -1:
        directory, moduleName = config.split('/')
        classModule = importlib.import_module(f"{package}.{directory}.{moduleName}")
    else:
        moduleName = config
        classModule = importlib.import_module(f"{package}.{moduleName}")
    classTokens = moduleName.split('_')
    className = ''.join(ele.title() for ele in classTokens)
    className = getattr(classModule, className)
    return className

if argument.train_files:

    if config['training']['trainfiles_axis'] not in ['index', 'column']:
        raise Exception('trainfiles_axis can only be either index or column')
    mapAxis = 0 if config['training']['trainfiles_axis'] == 'index' else 0

    def readFile(trainfile):
        df = pd.read_csv(f"data/{trainfile}.csv")
        if eval(config['preprocess']['groups']) and mapAxis == 0:
            df['group'] = trainfile
        return df

    df_train = pd.concat(list(map(readFile, argument.train_files)), axis = mapAxis)

    # Global preprocess training:
    for action in eval(config['preprocess']['preprocessors']):
        preprocessClass = getClassFromConfig('preprocess', action)
        preprocessObject = preprocessClass()
        df_train = preprocessObject.preprocess(df_train)

    # Features preprocess training:
    baseTransformer = getClassFromConfig('preprocess', 'base_transformer')()
    encoders = []
    transformers = eval(config['preprocess']['transformers'])
    # passthrough is a special transformer to keep original features untouched.
    if 'passthrough' in transformers:
        passthrough = getClassFromConfig('preprocess', 'passthrough')()
        features = df_train.columns if transformers['passthrough'] == [] else transformers['passthrough']
        encoders.append(('passthrough', passthrough.pipeline(), features))
        del transformers['passthrough']
    for action, features in transformers.items():
        preprocessClass = getClassFromConfig('preprocess', action)
        preprocessObject = preprocessClass()
        encoders.append((action, preprocessObject.pipeline(), features))

    preprocessor = ColumnTransformer(
        transformers=encoders,
        verbose_feature_names_out=eval(config['preprocess']['verbose_feature_names_out'])
    )

    labelName = config['preprocess']['label']
    y_train = df_train[labelName].to_frame(labelName)
    x_train = df_train
    x_train.drop(labelName, axis=1, inplace=True)

    labels = y_train
    if eval(config['preprocess']['label_encode']) == True:
        le = LabelEncoder()
        y_train = pd.DataFrame(list(le.fit_transform(y_train)), columns=[labelName])
        labels = le.classes_

    x_train = preprocessor.set_output(transform="pandas").fit_transform(x_train)

    if eval(config['preprocess']['verbose_feature_names_out']) and 'passthrough__group' in df_train.columns:
        x_train.drop('passthrough__group', axis=1, inplace=True)

    # Training plots:
    # @todo: Further filter features for eda plots.
    for plot in eval(config['eda']['plots']):
        plotClass = getClassFromConfig('plots', plot)
        plotObject = plotClass(config, x_train, y_train, labels, '/'.join(argument.train_files))
        plotObject.run()

    # Splitting training / validaton sets:
    # Only one splitting method by script execution.
    # Will return an iterable list of x,y tuple
    groups = None
    if config['training'].get('splitting_method') is not None:
        iteratorClass = getClassFromConfig('split', config['training']['splitting_method'])
    if config['training'].get('group_column') is not None:
        groups = x_train[config['training']['group_column']]
    if config['training'].get('nb_splits') is not None:
        nb_splits = int(config['training']['nb_splits'])
    else:
        nb_splits = len(argument.train_files)
    cv = iteratorClass.split(nb_splits, x_train, y_train, groups)

    # Assemble all estimators (sampling, classifiers ...) in a single pipeline
    factoredPipeline = BasePipeline(config)
    for estimator in eval(config['training']['estimators']):
        estimator = getClassFromConfig('pipeline', estimator)(config).getEstimator()
        factoredPipeline.addStep(estimator)
    pipeline = factoredPipeline.getPipeline()

    # Combine the pipeline and splits in cross_validate
    scores = cross_validate(pipeline, x_train, y_train, cv = cv)

    print(scores)
    # Optionnaly save models and coefs (feature importance) from the output of cross_validate
