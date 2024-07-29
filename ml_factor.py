import pandas as pd
import argparse
import configparser
import importlib
from sklearn.compose import ColumnTransformer

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

    # Global preprocess training:
    for action in eval(config['preprocess']['preprocessors']):
        preprocessClass = getClassFromConfig('preprocess', action)
        preprocessObject = preprocessClass()
        df_train = preprocessObject.preprocess(df_train)

    df_train = preprocessor.set_output(transform="pandas").fit_transform(df_train)

    groupColumn = 'passthrough__group' if eval(config['preprocess']['verbose_feature_names_out']) else 'group'
    if eval(config['preprocess']['groups']) and groupColumn in df_train.columns:
        df_train.drop(groupColumn, axis=1, inplace=True)

    # Training plots:
    for plot in eval(config['eda']['plots']):
        plotClass = getClassFromConfig('eda_plots', plot)
        plotObject = plotClass(df_train, config, '/'.join(argument.train_files))
        plotObject.run()

# cross_validate with cv from validate directory, dynamically
