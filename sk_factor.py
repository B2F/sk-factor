import argparse
import re
from src.engine.config import Config
from src.engine.preprocessors import Preprocessors
from src.engine.transfomers import Transformers
from src.engine.training import Training
from src.engine.plots import Plots
from src.engine.debug import Debugger
from src.engine.predictions import Predictions
from src.engine.plugins import Plugins

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", help = "Use a config file from the config/ directory", default='sk_factor')

# -t and -p arguments can be cumulated
parser.add_argument("-t", "--train_files", help = "Train with given file(s)", required = False, nargs = "*")
parser.add_argument("-p", "--predict_files", help = "Predict with given file(s)", required = False, nargs = "*")
parser.add_argument("-m", "--model_file", help = "Model file(s) used for predictions", required = False, nargs = "*")

parser.add_argument("-d", "--debug", help = "Enable debugging", action='store_true')

group = parser.add_mutually_exclusive_group()
group.add_argument("-ef", "--eda", help = "EDA plots only", action='store_true', required = False)
group.add_argument("-tf", "--train", help = "Training only", action='store_true', required = False)
group.add_argument("-pf", "--predict", help = "Predict only", action='store_true', required = False)

argument = parser.parse_args()

config = Config(argument.config)
reConfig = re.search(r"(?:.*/)?([^\/\.]*)(?:\.toml)$", argument.config)
config.set('dataset', 'filename', reConfig.group(1))

config.set('debug', 'enabled', True) if argument.debug else config.set('debug', 'enabled', False)

if argument.eda:
    config.set('eda', 'enabled', True)
    config.set('training', 'enabled', False)
    config.set('predictions', 'enabled', False)
elif argument.train:
    config.set('eda', 'enabled', False)
    config.set('training', 'enabled', True)
    config.set('predictions', 'enabled', False)
elif argument.predict:
    config.set('eda', 'enabled', False)
    config.set('training', 'enabled', False)
    config.set('predictions', 'enabled', True)

Debugger.attach(config)

trainfiles = argument.train_files if argument.train_files else config.get('dataset', 'files')

models = []

if trainfiles:

    ###
    # Step 0. Reading files from command line

    loader = config.get('dataset', 'loader')
    df_train = Plugins.create('loader', loader, config, trainfiles).load()

    ###
    # Step 1. Preprocessing

    x_train, y_train, labels = Preprocessors.encodeLabel(config, df_train)
    x_train = Transformers.apply(config, x_train)

    # Order below is significant if drop_rows if used with drop_rows_to_csv.
    # In that case only x_train is relevant if used as a predict_file.

    print('\nBefore Y preprocessing:')
    print(y_train.shape)

    y_train = Preprocessors.apply(config, y_train)

    print('\nBefore X preprocessing:')
    print(y_train.shape)

    x_train = Preprocessors.apply(config, x_train)

    ###
    # Step 2. EDA plots:

    if config.eq('eda', 'enabled', True):

        identifier = identifier = '/'.join(trainfiles) if len(trainfiles) > 1 else trainfiles
        Plots().run(config, x_train, y_train, labels, identifier)

    ###
    # Step 3. Training:

    if config.eq('training', 'enabled', True):

        models = Training(config, x_train, y_train, labels).run()

### Step 4. Predictions from model:

if config.eq('predictions', 'enabled', True):

    predict_files = argument.predict_files if argument.predict_files else [config.get('predictions', 'predict_file')]
    Predictions(config, predict_files, labels, models).run()

# @todo
# Add unit tests
# Model stacking
