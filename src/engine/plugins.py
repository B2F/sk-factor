import importlib
import re
from plugins.pipeline.base_estimator import BaseEstimator
from plugins.plots.base_report import Report
from plugins.preprocess.base_preprocessor import BasePreprocessor
from plugins.preprocess.base_transformer import BaseTransformer
from plugins.split.base_cv import BaseCv
from plugins.training.base_runner import BaseRunner
from plugins.loader.base_loader import BaseLoader

class Plugins():
    """ Plugins factory.
    """

    @staticmethod
    def create(package: str, module: str, *args):

        if module.find('/') != -1:
            directory, separator, moduleName = module.rpartition('/')
            package = package + '.' + directory.replace('/', '.')
            classModule = importlib.import_module(f"plugins.{package}.{moduleName}")
        else:
            moduleName = module
            classModule = importlib.import_module(f"plugins.{package}.{moduleName}")

        classTokens = moduleName.split('_')
        className = ''.join(ele.title() for ele in classTokens)
        className = getattr(classModule, className)

        object = className(*args)

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
