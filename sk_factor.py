import importlib
import argparse
from src.engine.config import Config
from src.engine.data import Data
from src.engine.preprocessors import Preprocessors
from src.engine.transfomers import Transformers
from src.engine.training import Training
from src.engine.plots import Plots

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", help = "Use a config file from the config/ directory", default='sk_factor')

# -t and -p arguments can be cumulated
parser.add_argument("-t", "--train_files", help = "Train with given file(s)", required = False, nargs = "*")
parser.add_argument("-p", "--test_files", help = "Predict with given file(s)", required = False, nargs = "*")
parser.add_argument("-m", "--model_file", help = "Model file(s) used for predictions", required = False, nargs = "*")

parser.add_argument("-d", "--debug", help = "Enable debugging", action='store_true')

argument = parser.parse_args()

config = Config(argument.config)

if argument.debug or config.eq('debug', 'enabled', True):
    debugpy = importlib.import_module("debugpy")
    debugpy.listen((config.get('debug', 'host'), config.get('debug', 'port')))
    if config.eq(*('debug', 'wait_for_client'), True):
        debugpy.wait_for_client()

if argument.train_files:

    ###
    # Step 0. Reading files from command line

    mergeAxis = config.get('training', 'trainfiles_axis')
    dataFrames = list(map(Data.readFile, argument.train_files))
    if config.eq(*('preprocess', 'groupFiles'), True):
        dataFrames = list(map(Data.addGroup, *(dataFrames, 'group')))
    df_train = Data.mergeFiles(dataFrames, mergeAxis)

    ###
    # Step 1. Preprocessing

    label = config.get('preprocess', 'label')

    preprocessors = config.get('preprocess', 'preprocessors')
    x_train, y_train = Preprocessors.apply(preprocessors, config, df_train, label)

    transformers = config.get('preprocess', 'transformers')
    many_to_one = config.get('preprocess', 'transformers_many_to_one')
    x_train = Transformers.apply(transformers, many_to_one, config, x_train)

    labels = y_train
    if config.eq('preprocess', 'label_encode', True):
        y_train, labels = Transformers.labelEncode(y_train, label)

    ###
    # Step 2. EDA plots:

    if config.eq('eda', 'show_plots', True) or config.eq('eda', 'save_images', True):

        plots = config.get('eda', 'plots')
        identifier = '/'.join(argument.train_files)
        Plots().run(plots, config, x_train, y_train, labels, identifier)

    ###
    # Step 3. Training:

    if config.eq('training', 'enabled', True):

        if type(config.get('training', 'nb_splits')) is int :
            nb_splits = config.get('training', 'nb_splits')
        else:
            nb_splits = len(argument.train_files)

        Training(config, x_train, y_train, labels, nb_splits).run()

    ### Step 4. Predictions from model:

    # @todo
    # Optionnaly save models and coefs (feature importance) from the output of cross_validate
