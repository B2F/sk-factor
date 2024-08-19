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

argument = parser.parse_args()

config = Config(argument.config)
reConfig = re.search(r"(?:.*/)?([^\/\.]*)(?:\.toml)$", argument.config)
config.set('dataset', 'filename', reConfig.group(1))

config.set('debug', 'enabled', True) if argument.debug else config.set('debug', 'enabled', False)

Debugger.attach(config)

trainfiles = argument.train_files if argument.train_files else config.get('dataset', 'files')

model = None

if trainfiles:

    ###
    # Step 0. Reading files from command line

    loader = config.get('dataset', 'loader')
    df_train = Plugins.create('loader', loader, config, trainfiles).load()

    ###
    # Step 1. Preprocessing

    x_train, y_train, labels = Preprocessors.apply(config, df_train)
    x_train = Transformers.apply(config, x_train)

    ###
    # Step 2. EDA plots:

    if config.eq('eda', 'show_plots', True) or config.eq('eda', 'save_images', True):

        identifier = identifier = '/'.join(trainfiles) if len(trainfiles) > 1 else trainfiles
        Plots().run(config, x_train, y_train, labels, identifier)

    ###
    # Step 3. Training:

    if config.eq('training', 'enabled', True):

        model = Training(config, x_train, y_train, labels).run()

### Step 4. Predictions from model:

if config.eq('predictions', 'enabled', True):

    predict_files = argument.predict_files if argument.predict_files else [config.get('predictions', 'predict_file')]
    Predictions(config, predict_files, labels, model).run()

# @todo
# Add unit tests
# Model stacking
