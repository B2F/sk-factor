import importlib
import re
from plugins.pipeline.base_estimator import BaseEstimator
from plugins.plots.base_report import Report
from plugins.preprocess.base_preprocessor import BasePreprocessor
from plugins.preprocess.base_selector import BaseSelector
from plugins.preprocess.base_transformer import BaseTransformer
from plugins.split.base_cv import BaseCv
from plugins.training.base_runner import BaseRunner
from plugins.loader.base_loader import BaseLoader

class Plugins():
    """ Plugins factory.
    """

    PACKAGE_BASE = 'plugins'

    @staticmethod
    def create(package: str, module: str, config = None, *args):

        # @todo allow to configure plugins retrieval from external package / directory.
        # If class not found default back to sf_factor base plugins directory.
        packageBase = Plugins.PACKAGE_BASE
        if config.get('dataset', 'plugins'):
            packageBase = config.get('dataset', 'plugins')

        if module.find('/') != -1:
            directory, separator, moduleName = module.rpartition('/')
            package = package + '.' + directory.replace('/', '.')
        else:
            moduleName = module

        # First, we try to find the plugin from toml's [dataset] plugins config if available:
        pluginPath = f"{Plugins.PACKAGE_BASE}.{package}.{moduleName}"
        try:
            classModule = importlib.import_module(f"{packageBase}.{pluginPath}")
        # Else, fallback to
        except ModuleNotFoundError:
            if Plugins.PACKAGE_BASE != packageBase:
                classModule = importlib.import_module(f"{pluginPath}")

        classTokens = moduleName.split('_')
        className = ''.join(ele.title() for ele in classTokens)
        className = getattr(classModule, className)

        object = className(config, *args)

        Plugins.checkPackageClass(object, package)

        return object

    @staticmethod
    def checkPackageClass(object, package):

        # Check validity relating to package.
        packagesClasses = {
            'loader': BaseLoader,
            'pipeline': BaseEstimator,
            'plots': Report,
            'preprocess.preprocessor': BasePreprocessor,
            'preprocess.selector': BaseSelector,
            'preprocess.transformer': BaseTransformer,
            'split': BaseCv,
            'training': BaseRunner,
            # 'predictions': BasePredictions,
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
