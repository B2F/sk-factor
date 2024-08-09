import os
import pandas as pd
import argparse
import configparser
import importlib
from sklearn.compose import ColumnTransformer
from pipeline.base_pipeline import BasePipeline
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help = "Use a config file from the config/ directory", default="sk_factor.ini")

# -t and -p arguments can be cumulated
parser.add_argument("-t", "--train_files", help = "Train with given file(s)", required = False, nargs = "*")
parser.add_argument("-p", "--test_files", help = "Predict with given file(s)", required = False, nargs = "*")
parser.add_argument("-m", "--model_file", help = "Model file(s) used for predictions", required = False, nargs = "*")

argument = parser.parse_args()

config = configparser.ConfigParser()
if argument.config != 'sk_factor.ini':
    if not os.path.isfile('config/' + argument.config):
        raise Exception('config/' + argument.config + 'not found.')
    config.read('config/' + argument.config)
else:
    config.read(argument.config)

if eval(config['debug']['enabled']) == True:
    debugpy = importlib.import_module("debugpy")
    debugpy.listen((config['debug']['host'], eval(config['debug']['port'])))
    if eval(config['debug']['wait_for_client']) == True:
        debugpy.wait_for_client()

# Modules loading
def getClassFromConfig(package, config):
    if config.find('/') != -1:
        directory, separator, moduleName = config.rpartition('/')
        package = package + '.' + directory.replace('/', '.')
        classModule = importlib.import_module(f"{package}.{moduleName}")
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
    print('Initial CSV shape:')
    print(df_train.shape)

    dfColumns = list(df_train.columns)

    transformers = eval(config['preprocess']['transformers'])
    if 'drop_columns' in transformers:

        for feature in transformers['drop_columns']:
            dfColumns.remove(feature)

        df_train = df_train[dfColumns]
        print('\nDropped columns:')
        print(df_train.shape)

    for action in eval(config['preprocess']['preprocessors']):
        preprocessClass = getClassFromConfig('preprocess', 'preprocessor/' + action)
        preprocessObject = preprocessClass(config, df_train)
        df_train = preprocessObject.transform()

    # Features preprocess training:
    encoders = []
    labelName = config['preprocess']['label']
    y_train = df_train[labelName].to_frame(labelName)
    x_train = df_train
    x_train.drop(labelName, axis=1, inplace=True)
    dfColumns.remove(labelName)

    if eval(config['preprocess']['groups']):
        dfColumns.remove('group')

    # passthrough is a special transformer to keep original features untouched.
    if 'passthrough' in transformers:
        passthrough = getClassFromConfig('preprocess', 'transformer/passthrough')(config, x_train)
        features = list(dfColumns) if transformers['passthrough'] == [] else transformers['passthrough']
        encoders.append(('passthrough', passthrough.pipeline(), features))
        del transformers['passthrough']
    for action, features in transformers.items():
        if action == 'drop_columns':
            continue
        preprocessClass = getClassFromConfig('preprocess', 'transformer/' + action)
        preprocessObject = preprocessClass(config, x_train)
        # Preprocess for columns selected in the .ini config.
        features = list(dfColumns) if features == [] else features
        encoders.append((action, preprocessObject.pipeline(), features))

    preprocessor = ColumnTransformer(
        transformers=encoders,
        verbose_feature_names_out=eval(config['preprocess']['verbose_feature_names_out'])
    )

    labels = y_train
    if eval(config['preprocess']['label_encode']) == True:
        le = LabelEncoder()
        y_train = pd.DataFrame(list(le.fit_transform(y_train.values.ravel())), columns=[labelName])
        labels = le.classes_

    x_train = preprocessor.set_output(transform="pandas").fit_transform(x_train)

    if eval(config['preprocess']['verbose_feature_names_out']) and 'passthrough__group' in df_train.columns:
        x_train.drop('passthrough__group', axis=1, inplace=True)

    # Training plots:
    if eval(config['eda']['show_plots']) is True or eval(config['eda']['save_images']) is True:
        for plot in eval(config['eda']['plots']):
            plotClass = getClassFromConfig('plots', plot)
            plotObject = plotClass(config, x_train, y_train, labels, '/'.join(argument.train_files))
            plotObject.run()

    if eval(config['training']['enabled']) is True:

        # Splitting training / validaton sets:
        # Only one splitting method by script execution.
        # Will return an iterable list of x,y tuple
        groups = None
        iteratorClass = None
        if config['training'].get('splitting_method') is not None:
            iteratorClass = getClassFromConfig('split', config['training']['splitting_method'])
        if config['training'].get('group_column') is not None and config['training']['group_column'] in x_train.index:
            groups = x_train[config['training']['group_column']]
        if config['training'].get('nb_splits') is not None:
            nb_splits = int(config['training']['nb_splits'])
        else:
            nb_splits = len(argument.train_files)
        if nb_splits > 1 and iteratorClass is not None:
            iteratorObject = iteratorClass(config, x_train, y_train, nb_splits, groups)
            cv = iteratorObject.split()
        else:
            cv = nb_splits

        # Assemble all estimators (sampling, classifiers ...) in a single pipeline
        factoredPipeline = BasePipeline(config)
        for estimator in eval(config['training']['estimators']):
            estimator = getClassFromConfig('pipeline', estimator)(config, x_train, y_train).getEstimator()
            factoredPipeline.addStep(estimator)
        pipeline = factoredPipeline.getPipeline()

        for runner in eval(config['training']['runners']):
            plotClass = getClassFromConfig('training', runner)
            plotObject = plotClass(config, x_train, y_train, labels, 'training test')
            plotObject.run(pipeline, cv)

        # @todo: add RFECV features optimisation option

        # @todo: Add an additionnal step for TunedThresholdClassifierCV, GridSearchCV, RandomizedSearchCV and such.

        # cv_results = cross_validate(pipeline, x_train, y_train, cv = cv, return_estimator=True)
        # print(cv_results)

        # @todo: set the scroring in a separate modulable package, with pre-made class for confusion matrix and precision recall

        # @todo try normalize='true to better understand classes imbalance

        # Optionnaly save models and coefs (feature importance) from the output of cross_validate

    # @todo add a --gui cli argument with automatic plugins (classes options) discovery
    # The GUI would have tabs corresponding to following .ini sections (discovered dynamically, possibility to add sections for custom plugins)
