import importlib
import re
from plugins.estimators.base_estimator import BaseEstimator
from plugins.plots.base_report import Report
from plugins.preprocess.base_preprocessor import BasePreprocessor
from plugins.preprocess.base_selector import BaseSelector
from plugins.preprocess.base_transformer import BaseTransformer
from plugins.split.base_cv import BaseCv
from plugins.training.base_runner import BaseRunner
from plugins.loader.base_loader import BaseLoader
from plugins.predictions.base_predictor import BasePredictor

class Plugins():
    """ Plugins factory.
    """

    PACKAGE_BASE = 'plugins'

    @staticmethod
    def create(package: str, module: str, config = None, *args):

        config_plugins = config.get('dataset', 'plugins')

        # Handle both single plugin package (string) and multiple plugin packages (list/array)
        if config_plugins:
            if isinstance(config_plugins, str):
                # Single plugin package (backward compatibility)
                packageBases = [Plugins.PACKAGE_BASE, config_plugins]
            elif isinstance(config_plugins, list):
                # Multiple plugin packages
                packageBases = [Plugins.PACKAGE_BASE] + config_plugins
            else:
                raise Exception('dataset plugins must be either list or str')
        else:
            packageBases = [Plugins.PACKAGE_BASE]

        # If the module string contains a path, split directory from module name.
        if module.find('/') != -1:
            directory, separator, moduleName = module.rpartition('/')
            package = package + '.' + directory.replace('/', '.')
        else:
            moduleName = module

        # First, we try to find the plugin from toml's [dataset] plugins config if available:
        pluginPath = f"{Plugins.PACKAGE_BASE}.{package}.{moduleName}"

        classModule = None
        # Try each plugin package in order
        for pkg_base in packageBases:
            try:
                classModule = importlib.import_module(f"{pkg_base}.{pluginPath}")
                break # Found the plugin, exit the loop
            except ModuleNotFoundError:
                continue

        # If not found in any custom packages, fall back to default sk_factor plugins directory:
        if classModule is None:
            try:
                classModule = importlib.import_module(f"{pluginPath}")
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f"Plugin '{module}' not found in any of the specified plugin packages: {packageBases} or default plugins")

        classTokens = moduleName.split('_')
        className = ''.join(ele.title() for ele in classTokens)
        className = getattr(classModule, className)

        object = className(config, *args)

        Plugins.checkPackageClass(object, package)

        return object

    @staticmethod
    def checkPackageClass(object, package):

        # Check if package class is valid:
        packagesClasses = {
            'loader': BaseLoader,
            'estimators': BaseEstimator,
            'plots': Report,
            'preprocess.preprocessor': BasePreprocessor,
            'preprocess.selector': BaseSelector,
            'preprocess.transformer': BaseTransformer,
            'split': BaseCv,
            'training': BaseRunner,
            'predictions': BasePredictor,
        }

        matchingPackage = ''
        for regularPackage in packagesClasses:
            if not re.match(regularPackage, package):
                continue
            else:
                matchingPackage = regularPackage
                break

        if not matchingPackage:
            raise Exception(f'Plugin''s package {package} does not exist.')

        expectedClass = packagesClasses[matchingPackage]
        if not isinstance(object, expectedClass):
            raise Exception(f'Plugin of package {package} must be of class {expectedClass}.')
