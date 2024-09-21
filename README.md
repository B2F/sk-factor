<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/B2F/sk-factor/blob/main/images/logo.png">
    <img src="https://github.com/B2F/sk-factor/blob/main/images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">SK Factor: one liner CLI to factor and reuse code from sklearn projects</h3>

  <p align="center">
    scikit-learn (sklearn) streamlined workflow and command line interface
    <br />
    <!-- <a href="https://github.com/B2F/sk-factorBest-README-Template"><strong>Explore the docs »</strong></a> -->
    <br />
    <a href="https://github.com/B2F/sk-factor/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/B2F/sk-factor/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#what-is-sk-factor-">What is SK Factor ?</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage-examples">Usage examples</a>
      <ul>
        <li><a href="#binary-target-credit_card_fraudtoml">Binary targets</a></li>
        <li><a href="#multiclass-target">Multiclass target</a></li>
        <li><a href="#regression-target">Regression target</a></li>
      </ul>
    </li>
    <li>
      <a href="#configuration-sections">Configuration sections</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#preprocess">Preprocess</a></li>
        <li><a href="#eda">Exploratory Data Analysis</a></li>
        <li><a href="#training-1">Training</a></li>
        <li><a href="#predictions">Predictions</a></li>
      </ul>
    </li>
    <li><a href="#plugin-system">Plugin system</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## What is SK Factor ?

```bash
python sk_factor.py -c examples/open_ml/config/credit_card_fraud.toml
```

SK Factor is *a framework and CLI used to factor scikit-learn repetitive and common developement tasks** so you can quickly access and reuse them.

Inspired by module based software projects, SK Factor's goal is to streamline the process of developing machine learning projects with sklearn,

Many of the development **tasks are redundant when building a sklearn project**, especially when it comes to **pipelines**.

From the get go (in a single command line run), you will be able to do many advanced actions:

- Unlimited split training in one run
- Train on different estimators (classifiers, regressors) for each splits
- Compare predictions with different models, at once
- Easily save and export models, plots
- Standardize and readily available report / scoring templates

  (roc curve, classification report, confusion matrix, precision recall, features permutation ...)
- Enable and disable any step you choose (sampler, transformer)

To achieve this, SK Factor uses **.toml configuration files** reflecting each part of the **workflow**:

<img src="https://github.com/B2F/sk-factor/blob/main/images/workflow.png" alt="SK Factor workflow">

Each steps can be customized via a convenient [plugins system](#plugin-system) (loader, estimators ...)

By default, running sk_factor.py will run the complete workflow specified in your **.toml** configuration file.
You can choose to filter only one part of the workflow using one of those arguments:

* **--explore** : preprocess the data and display plots (uses only [**dataset**](#dataset), preprocess and [**eda config**](#eda))
* **--train**   : train on given estimators and display plots (uses only [**dataset**](#dataset), preprocess and [**training config**](#training))
* **--predict** : predicts with chosen model(s) and display reports
  (uses only **dataset**, preprocess and **predictions config**)

See practical **.toml configuration** [examples below](#usage-examples)

<!-- USAGE EXAMPLES -->
## Usage examples

### Binary target

```sh
python sk_factor.py -c examples/open_ml/config/credit_card_fraud.toml
```

The [credit_card_fraud.toml](https://github.com/B2F/sk-factor/blob/main/examples/open_ml/config/credit_card_fraud.toml) config file predicts credit card frauds using open_ml's CreditCardFraudDetection dataset.

#### Preprocessing:
```toml
[preprocess]
# preprocessing is always required
````

* The dataset is scaled, shuffled, and 'Time' column is removed.
```toml
preprocessors.shuffle = 1
transformers.scaler = []
preprocessors.drop_columns = ['Time']
```

* 25000 rows are removed at the end of the dataset and used for predictions.

```toml
preprocessors.drop_rows = -25000
drop_rows_to_predict_file = true
```

#### EDA:
```toml
[eda]
enabled = true
```
* The target distribution is displayed

``` toml
show_plots = true
plots = [
  'distribution_y',
]
```

#### Training:
```toml
[training]
enabled = true
```

* The dataset imbalance is mitigated with the tomek links algorithm.

```toml
pipeline = 'imblearn.pipeline'
samplers = [
    'sampler/tomek_links',
]
```

* A kfold_stratified splitting is done 3 times.

```toml
splitting_method.kfold_stratified = 3
```

* Two estimators are used on each splits: kneighbors_classifier and sgd_classifier

```toml
estimators = [
    'classifier/kneighbors_classifier',
    'classifier/sgd_classifier',
]
```

* Training score is f1.

```toml
runners = [
    'score',
]
scoring = 'f1'
```

* models are written to the models directory

```toml
save_model = true
model_timestamp = false
models_directory = 'models'
```

#### Predictions

```toml
[predictions]
objective = 'binary'
loader = 'csv'
preprocess = false
enabled = true
```

* Uses models generated at the training phase with a threshold of 0,5

```toml
models = [
  'models/credit_card_fraud-classifier/kneighbors_classifier.pkl',
  'models/credit_card_fraud-classifier/sgd_classifier.pkl'
]
threshold = 0.5
```

* Predictions are saved to .csv files for each model and displayed in the console

```toml
predict_file = 'tests/credit_card_fraud/test.csv'
predictions_directory = 'tests/credit_card_fraud/predictions'
save_predictions = true
predictions_timestamp = false
# Keep original features data features in the final predictions columns.
keep_data = false
```

### Multiclass target

```sh
python sk_factor.py -c examples/toy_datasets/config/iris.toml
```

The [iris.toml](https://github.com/B2F/sk-factor/blob/main/examples/toy_datasets/config/iris.toml) config file extracts data from sklearn toy datasets. This config files predicts a plant class from sepal attributes.

#### Preprocessing:
* The dataset is shuffled
* 5 rows are removed at the end for preditctions

#### EDA:
* A pair plot and and heatmap are displayed
* DPI resolution is set to 200

#### Training:
* The dataset imbalance is mitigated with near miss and yeo johnson algorithms
* Two kfold_shuffle and two kfold_stratified splits are made
* Estimators linear svc, xgboost and lgbm random forest are used on each split
* Accuracy score is calculated and printed

#### Predictions
* Uses sequential model data from current script execution training file
* Predictions are saved to .csv files for each model and displayed in console

### Regression target

```sh
python sk_factor.py -c examples/open_ml/config/happiness_rank.toml
```

The [happiness_rank.toml](https://github.com/B2F/sk-factor/blob/main/examples/open_ml/config/happiness_rank.toml) config file extracts data from open_ml and predicts a happiness score based on demographics and lifestyle attributes.

#### Preprocessing:
* Passthrough 7 attributes ('Economy', 'Family', 'Health', 'Freedom', ...)
* Applies one hot encoder to Region
* Scales the 'Standard Error' column
* Shuffles dataset and keep 5 rows at the end for predictions

#### EDA:
* Plots the heatmap with 35*35 and 200 DPI

#### Training:
* Create 5 kfold and 5 shuffled kfold splits
* Applies xgboost, lgbm regressor and hgbr estimator on each splits
* Print the r2 score for each estimator
* Save model files

#### Predictions
* Predicts from previously saved model files

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting started

SK Factor is a standard Python OOP script organized in packages and modules.

By default, running sk_factor requires the following dependencies:

### Prerequisites

  * **[sklearn][sklearn]**    -> core functionality such as pipeline is based on sklearn
  * **[imblearn][imblearn]**   -> provides advanced samplers to mitigate dataset unbalance
  * **[toml][toml]**       -> standard format for configuration file
  * **[argparse][argparse]**   -> provides CLI arguments handling
  * **[pandas][pandas]**     -> advanced dataset arrays operations
  * **[matplotlib][matplotlib]** -> data visualisation
  * **[seaborn][seaborn]**    -> diagrams plots

The standard python package installer PIP is required:

  ```sh
  pip install sklearn imblearn toml argparse pandas matoplotlib seaborn
  ```

Additionnal dependencies:

  * [lightgbm][lightgbm] -> additionnal gradient boost estimators
  * [xgboost][xgboost]  -> additionnal gradient boost estimators
  * [shap][shap]     -> advanced features analysis (such as permutation)
  * [openml][openml]   -> access to machine learning datasets (instead of csv)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/B2F/sk-factor.git
   ```
2. Grab one of the [examples below](#usage-examples)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration sections

### [dataset]

The **[dataset]** section is used to describe the data source and how to parse it.

Example from [happiness_rank.toml](https://github.com/B2F/sk-factor/blob/main/examples/open_ml/config/happiness_rank.toml):

```toml
[dataset]
loader = 'open_ml'
files = ['HappinessRank_2015']
show_columns = true
plugins = 'examples.open_ml'
```

#### loader
Data parser from which one or multiple files are read.

Options: [**csv**](https://github.com/B2F/sk-factor/blob/main/plugins/loader/csv.py'), [**open_ml**](https://github.com/B2F/sk-factor/blob/main/examples/open_ml/plugins/loader/open_ml.py), [**toy_datasets**](https://github.com/B2F/sk-factor/blob/main/examples/toy_datasets/plugins/loader/toy_datasets.py).

#### files
Array of arguments to be passed to the loader.

Can be replaced by --train_files CLI argument.

#### show_columns

Will display all available dataset's columns at the beginning of the CLI output.

#### plugins
Package or directory used to override plugins definition. @see [plugins system](#plugin-system).

### [preprocess]
The **preprocess** section is used to apply transformation to the dataset (drop, shuffle, encode, passthrough).

#### label
Column name used as the target label.

#### label_encode
Boolean, specify if the label must be encoded (use true for string).

#### transformers.passthrough
List of columns left unchanged, use an empty [] for all.

#### transformers.one_hot_encoder
Encode values in a new categorical column.

#### transformers.ordinal_encoder
Encode values by replacing them in the same column.

#### transformers.scaler
Applies sklearn StandardScaler.

#### transformers.shuffle
Shuffles the DataFrame with the random state as value.

#### preprocessors.drop_rows
Drops n rows at the beginning (positive integer), or from the end (negative integer).

#### drop_rows_to_predict_file
Use dropped rows for predictions (@see [predictions](#predictions) )

#### verbose_feature_names_out
Set to false to remove suffixes from one hot encoder

#### files_axis
Choose the dataframe merge axis when using multiple files

### [eda]
The **eda** section (Exploratory Data Analysis) is used with matplotlib and seaborn plots or anything else printed with Python.

#### enabled
If the EDA phase is skipped, no output, no file save (default: true).

Use the **--explore** CLI option to filter script execution on **eda** config only.

#### show_plots
To skip diagram or printed output, use show_plots = false.

#### save_images
Write plots visual to files.

#### save_timestamp
Append a timestamp suffix to saved files.

#### images_extension
Extension of saved files.

#### images_directory
Directory of saved plot files.

#### plots
Plot plugins to use, @see [plugins/plots](https://github.com/B2F/sk-factor/blob/main/plugins/plots)

Options: [**heatmap**](https://github.com/B2F/sk-factor/blob/main/plugins/plots/heatmap.py'), [**pairplot**](https://github.com/B2F/sk-factor/blob/main/examples/open_ml/plugins/plots/pairplot.py), [**distribution_y**](https://github.com/B2F/sk-factor/blob/main/plugins/plots/distribution_y.py), [**distribution_x**](https://github.com/B2F/sk-factor/blob/main/plugins/plots/distribution_x.py)

#### features
Specify an array of columns name to be used with the plot plugin above.

#### figsize
Figure size width and height in inches.

#### dpi
Figure resolution in DPI.

### [training]
The **training** section is used to train on splits and to to create models

#### enabled
If you want to force skip the training section, set to false (default: true)

Use the **--train** option to filter script execution on **training** config only.

#### pipeline
Training's pipeline module (default: 'imblearn.pipeline')

#### samplers
Imblearn samplers ('sampler/smote', 'sampler/tomek_links')

#### estimators
Classifier and regressor estimators (see plugins/classifiers).
```toml
# Ex:
estimators = [
    'classifier/logistic_regression',
    'classifier/ridge_classifier',
    'classifier/kneighbors_classifier',
    'classifier/sgd_classifier',
    'classifier/lgbm_classifier',
]
```

#### runners
Training score runners: 'score', 'classification_report', 'confusion_matrix', 'precision_recall' ... (@see plugins/training)

#### scoring
Scoring metric passed as argument to the score runner plugin ('f1', 'r2' ...)

#### splitting_method.kfold_stratified
Specify unlimited amount of sklearn splitting methods (one per line), with value as n_splits.
Ex:
```toml
splitting_method.kfold = 5
splitting_method.kfold_shuffle = 5
```

#### save_model
Save trained model

#### model_timestamp
Append timestamp suffix to model filename

#### models_directory
Saved models directory

### [predictions]

The **predictions** section is used to predict from training data or model files.

#### enabled = true
Enable or disable the **predictions** section section altogether.

Use the **--predict** option to filter CLI execution on the **predictions** section only.

#### loader = 'csv'
The loader plugin used to retrieve data for prediction.

#### preprocess
Choose weither or not to re-use the preprocessing section rules for the prediction data.
If you used preprocessors.drop_rows with drop_rows_to_predict_file enabled in the **preprocess** section, then your prediction data is already preprocessed and you'll want to set preprocess = false

#### predict_file
Path used to write predictions. If you set drop_rows_to_predict_file = true, then this file will be written with the number of rows from the original dataset, specified in preprocessors.drop_rows

#### models
An array of models files to use for predictions

#### objective
Options 'binary', 'multiclass', 'regressor'

#### threshold
Threshold parameter passed to the objective plugin to filter probabilities.

#### save_predictions
If set to true, predictions will be saved to the predictions_directory.

#### predictions_directory
Where to save predictions.

#### predictions_timestamp
Set to true to append the timestamp to predictions filenames.

#### keep_data
Set to true to keep all prediction's data columns in predictions files.

### [debug]

#### enabled
Enable python CLI debugger

#### port
Debug port (Usually 5678)

#### host
Host address (Usually '127.0.0.1')

#### wait_for_client
Set to true to start debugger with execution

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Plugin system

Default plugins are located in the [plugins](https://github.com/B2F/sk-factor/blob/main/plugins) directory:

* plugins/estimators
* plugins/loader
* plugins/plots
* plugins/predictions
* plugins/split
* plugins/training

You can override or add more functionnality by putting your plugin class files in a package containing a plugins/ directory, which hierarchy reflects the project's base [plugins](https://github.com/B2F/sk-factor/blob/main/plugins) structure.

This package is specified by the **plugins** key in your toml config's dataset section.

From the **[examples/toy_datasets/config/iris.toml](https://github.com/B2F/sk-factor/blob/main/examples/toy_datasets/config/iris.toml)** file:

```toml
[dataset]
...
plugins = 'examples.toy_datasets'
```

If you look into **[examples/toy_datasets/plugins](https://github.com/B2F/sk-factor/blob/main/examples/toy_datasets/plugins)** you will find a custom plugins structure.

<!-- ROADMAP -->
## Roadmap

- [x] clean OOP code architecture with a plugin system
- [x] custom loader
- [x] exploratory data analysis
- [x] training
- [x] predictions
- [ ] Sphinx documentation (complete list of configuration options in the external documentation)
- [ ] Manage default values for unspecified config elements
- [ ] Additionnal plugins (roc curve with threshold display on both roc and precision / recall)
- [ ] Stacking estimators
- [ ] Example with time series forecast with tsfresh and sktime
- [ ] More example use cases and tests
- [ ] Using src.engine to build a GUI, dynamic access to plugins

See the [open issues](https://github.com/B2F/sk-factor/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can help with following tags:

* plugins
* estimators
* loader
* plots
* predictions
* split
* training
* engine

Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

B2F - [Linkedin](https://www.linkedin.com/in/didier-boff-ba6683118/)

Project Link: https://github.com/B2F/sk-factor

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

SK Factor is relying on all those awesome Data Science projects !

  * Python
    * [toml][toml]
    * [argparse][argparse]
    * [pandas][pandas]
  * Visualization
    * [matplotlib][matplotlib]
    * [seaborn][seaborn]
  * Machine learning
    * [sklearn][sklearn]
    * [imblearn][imblearn]
    * Models
      * [lightgbm][lightgbm]
      * [xgboost][xgboost]
    * Features exploration
      * [shap][shap]
    * Datasets
      * [openml][openml]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[pandas]: https://pandas.pydata.org/
[seaborn]: https://seaborn.pydata.org/
[matplotlib]: https://matplotlib.org/
[sklearn]: https://scikit-learn.org/stable/
[toml]: https://toml.io/en/
[imblearn]: https://imbalanced-learn.org/stable/
[shap]: https://shap.readthedocs.io/en/latest/#
[lightgbm]: https://lightgbm.readthedocs.io/en/stable/
[xgboost]: https://xgboost.readthedocs.io/
[openml]: https://www.openml.org/
[argparse]: https://docs.python.org/3/library/argparse.html
