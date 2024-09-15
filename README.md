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

  <h3 align="center">SK Factor: one liner CLI to factor and reuse code from sklearn project</h3>

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

SK Factor aim to streamline the process of developing machine learning projects with sklearn,
it is a machine learning framework inspired by module based software projects.

Many of the development **tasks are redundant when building a sklearn project**, especially when it comes to **pipelines**.

SK Factor acts like **a frame of reference to factor scikit-learn repetitive and common developement tasks** so you can quickly access and reuse them.

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

* **--explore** : preprocess the data and display plots (uses only **dataset**, preprocess and **eda config**)
* **--train**   : train on given estimators and display plots (uses only **dataset**, preprocess and **training config**)
* **--predict** : predicts with chosen model(s) and display reports
  (uses only **dataset**, preprocess and **predictions config**)

See practical **.toml configuration** [examples below](#usage-examples)

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

<!-- USAGE EXAMPLES -->
## Usage examples

### Binary target (credit_card_fraud.toml)
This project example predicts credit card frauds using open_ml's CreditCardFraudDetection dataset.

```sh
python sk_factor.py -c examples/open_ml/config/credit_card_fraud.toml
```

#### Preprocessing:
* The dataset is scaled, shuffled, and 'Time' column is removed.
* 25000 rows are removed at the end of the dataset and used for predictions.

#### EDA:
* The target distribution is displayed

#### Training:
* The dataset imbalance is mitigated with the tomek links algorithm.
* A kfold_stratified splitting is done 3 times.
* Two estimators are used on each splits: kneighbors_classifier and sgd_classifier
* Training score is f1 and models are written to the models directory.

#### Predictions
* Uses models generated at the training phase
* Threshold is set to 0,5
* Predictions are saved to a file and displayed in the console

### Multiclass target

@todo

```sh
python sk_factor.py -c examples/toy_datasets/config/iris.toml
```

### Regression target
-> happiness_rank

@todo

```sh
python sk_factor.py -c examples/open_ml/config/happiness_rank.toml
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration sections

### [dataset]

The [dataset] section is used to describe the data source and how to parse it

Example from [happiness_rank.toml](https://github.com/B2F/sk-factor/tree/main/examples/open_ml/config/happiness_rank.toml):

```toml
[dataset]
loader = 'open_ml'
files = ['HappinessRank_2015']
show_columns = true
plugins = 'examples.open_ml'
```

#### loader
Data parser from which one or multiple files are read.

Options: [**csv**](https://github.com/B2F/sk-factor/tree/main/plugins/loader/csv.py'), [**open_ml**](https://github.com/B2F/sk-factor/tree/main/examples/open_ml/plugins/loader/open_ml.py), [**toy_datasets**](https://github.com/B2F/sk-factor/tree/main/examples/toy_datasets/plugins/loader/toy_datasets.py).

#### files
Array of arguments to be passed to the loader.

Can be replaced by --train_files CLI argument.

#### show_columns

Will display all available dataset's columns at the beginning of the CLI output.

#### plugins
Package or directory used to override plugins definition. @see [plugins system](#plugin-system).

### [preprocess]
Preprocessing options to apply transformation to the dataset (drop, shuffle, encode, passthrough).

Example from [happiness_rank.toml](https://github.com/B2F/sk-factor/tree/main/examples/open_ml/config/happiness_rank.toml):

```toml
[preprocess]
label = 'Happiness Score'
label_encode = false
transformers.passthrough = [
  'Economy (GDP per Capita)',
  'Family',
  'Health (Life Expectancy)',
  'Freedom',
  'Trust (Government Corruption)',
  'Generosity',
  'Dystopia Residual'
]
transformers.one_hot_encoder = ['Region']
transformers.scaler = ['Standard Error']
preprocessors.shuffle = 1
preprocessors.drop_rows = -5
drop_rows_to_predict_file = true
```

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

### [eda]
Options for Exploratory Data Analysis with matplotlib and seaborn plots or anything else printed with Python

Example from [happiness_rank.toml](https://github.com/B2F/sk-factor/tree/main/examples/open_ml/config/happiness_rank.toml):

```toml
[eda]
enabled = false
show_plots = true
save_images = false
save_timestamp = true
images_extension = 'png'
images_directory = 'output/eda'
plots = [
  'heatmap',
  'pairplot',
]
# distribution_x=column_a
figsize = [35, 35]
dpi = 200
# features = ['column a', 'column b', 'column n']
```

#### enabled
If the EDA phase is skipped altogether (no output, no file save).

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
Plot plugins to use, @see [plugins/plots](https://github.com/B2F/sk-factor/tree/main/plugins/plots)

Options: [**heatmap**](https://github.com/B2F/sk-factor/tree/main/plugins/plots/heatmap.py'), [**pairplot**](https://github.com/B2F/sk-factor/tree/main/examples/open_ml/plugins/plots/pairplot.py), [**distribution_y**](https://github.com/B2F/sk-factor/tree/main/plugins/plots/distribution_y.py), [**distribution_x**](https://github.com/B2F/sk-factor/tree/main/plugins/plots/distribution_x.py)

#### features
Specify an array of columns name to be used with the plot plugin above.

#### figsize
Figure size width and height in inches.

#### dpi
Figure resolution in DPI.

### [training]
-> split, power transforms (yeo_johnson), sample (nearmiss, smote, tomek links), classify and display score

@todo: explain

### [predictions]
-> load models, adjust threshold, display prediction tables for each models

@todo: explain

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Plugin system

Default plugins are located in the [plugins](https://github.com/B2F/sk-factor/tree/main/plugins) directory:

* plugins/estimators
* plugins/loader
* plugins/plots
* plugins/predictions
* plugins/split
* plugins/training

You can override or add more functionnality by putting your plugin class files in a package containing a plugins/ directory, which hierarchy reflects the project's base [plugins](https://github.com/B2F/sk-factor/tree/main/plugins) structure.

This package is specified by the **plugins** key in your toml config's dataset section.

From the **[examples/toy_datasets/config/iris.toml](https://github.com/B2F/sk-factor/tree/main/examples/toy_datasets/config/iris.toml)** file:

```toml
[dataset]
...
plugins = 'examples.toy_datasets'
```

If you look into **[examples/toy_datasets/plugins](https://github.com/B2F/sk-factor/tree/main/examples/toy_datasets/plugins)** you will find a custom plugins structure.

<!-- ROADMAP -->
## Roadmap

- [x] clean OOP code architecture with a plugin system
- [x] custom loader
- [x] exploratory data analysis
- [x] training
- [x] predictions
- [ ] Sphinx documentation (complete list of configuration options in the external documentation)
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
