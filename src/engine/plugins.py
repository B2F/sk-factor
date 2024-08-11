import importlib

class Plugins():

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

        return className(*args)
