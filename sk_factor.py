import importlib
import argparse
from src.engine.config import Config
from src.engine.data import Data
from src.engine.preprocessors import Preprocessors
from src.engine.transfomers import Transformers
from src.engine.plugins import Plugins
from src.engine.training import Training
from src.engine.plots import Plots

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", help = "Use a config file from the config/ directory", default='sk_factor')

# -t and -p arguments can be cumulated
parser.add_argument("-t", "--train_files", help = "Train with given file(s)", required = False, nargs = "*")
parser.add_argument("-p", "--test_files", help = "Predict with given file(s)", required = False, nargs = "*")
parser.add_argument("-m", "--model_file", help = "Model file(s) used for predictions", required = False, nargs = "*")

argument = parser.parse_args()

configObj = Config(argument.config)
config = configObj.getConfig()

if configObj.eq('debug', 'enabled', True):
    debugpy = importlib.import_module("debugpy")
    debugpy.listen((configObj.get('debug', 'host'), configObj.get('debug', 'port')))
    if configObj.eq(*('debug', 'wait_for_client'), True):
        debugpy.wait_for_client()

if argument.train_files:

    ###
    # Step 0. Reading files from command line

    mergeAxis = configObj.get('training', 'trainfiles_axis')
    dataFrames = list(map(Data.readFile, argument.train_files))
    if configObj.eq(*('preprocess', 'groupFiles'), True):
        dataFrames = list(map(Data.addGroup, *(dataFrames, 'group')))
    df_train = Data.mergeFiles(dataFrames, mergeAxis)

    ###
    # Step 1. Preprocessing

    transformers = configObj.get('preprocess', 'transformers')
    preprocessors = configObj.get('preprocess', 'preprocessors')

    df_train = Preprocessors.apply(preprocessors, df_train)

    dfColumns = list(df_train.columns)
    labelName = configObj.get('preprocess', 'label')
    y_train = df_train[labelName].to_frame(labelName)
    x_train = df_train
    x_train.drop(labelName, axis=1, inplace=True)
    dfColumns.remove(labelName)

    if configObj.get('preprocess', 'groups'):
        dfColumns.remove('group')

    x_train = Transformers.apply(transformers, x_train, configObj)

    labels = y_train
    if configObj.eq('preprocess', 'label_encode', True):
        y_train, labels = Transformers.labelEncode(y_train, labelName)

    ###
    # Step 2. EDA plots:

    if configObj.eq('eda', 'show_plots', True) or configObj.eq('eda', 'save_images', True):

        plots = configObj.get('eda', 'plots')
        identifier = '/'.join(argument.train_files)
        Plots().run(plots, config, x_train, y_train, labels, identifier)

    ###
    # Step 3. Training:

    if configObj.eq('training', 'enabled', True):

        estimators = configObj.get('training', 'estimators')
        runners = configObj.get('training', 'runners')

        if type(configObj.get('training', 'nb_splits')) is int :
            nb_splits = configObj.get('training', 'nb_splits')
        else:
            nb_splits = len(argument.train_files)

        split_method = configObj.get('training', 'splitting_method')
        group_column = configObj.get('training', 'group_column')

        Training(
            x_train,
            y_train,
            estimators,
            runners,
            config,
            labels,
            nb_splits,
            split_method,
            group_column).run()

    ### Step 4. Predictions from model:

    # Optionnaly save models and coefs (feature importance) from the output of cross_validate
