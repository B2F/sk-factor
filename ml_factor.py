import pandas as pd
import argparse
import configparser
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('input_csv', help = "CSV file to perform an analysis on")
parser.add_argument("-c", "--config", help = "Use a config file (cli args take precedence, similar keys)", default="ml_factor.ini")
argument = parser.parse_args()

# -t and -p arguments can be cumulated
parser.add_argument("-t", "--train", help = "Train a model, given the .ini [training] config", required = False, nargs = "*")
parser.add_argument("-p", "--predict", help = "Predict from settings given the .ini [predictions] config", required = False, nargs = "*")

df = pd.read_csv(f"data/{argument.input_csv}.csv")
print('\nCSV shape:')
print(df.shape)

config = configparser.ConfigParser()
config.read(argument.config)

if eval(config['DEFAULT']['debug']) == True:
    debugpy = importlib.import_module("debugpy")
    debugpy.listen(('127.0.0.1', 5678))
    debugpy.breakpoint()

# Features filtering:
includedFeatures = eval(config['features']['includedFeatures'])
columns = includedFeatures if includedFeatures != None else df.columns
droppedFeatures = eval(config['features']['droppedFeatures'])
columns = droppedFeatures if droppedFeatures != None else columns

df = df[columns]

# Auto pre-processing:
if eval(config['preprocess']['drop_na']) == True:
    df = df[columns].dropna(axis=0)

for plot in eval(config['eda']['plots']):
    if plot.find('/') != -1:
        directory, plot = plot.split('/')
        plotModule = importlib.import_module(f"eda_plots.{directory}.{plot}")
    else:
        plotModule = importlib.import_module(f"eda_plots.{plot}")
    plotClassTokens = plot.split('_')
    plotClassName = ''.join(ele.title() for ele in plotClassTokens)
    plotClass = getattr(plotModule, plotClassName)
    plotObject = plotClass(df, config, argument.input_csv)
    plotObject.run()
