import configparser
import os

class Config():

    _DEFAULT_DIRECTORY = 'config/'
    _config: dict

    def __init__(self, filename = 'sk_factor.ini'):

        if filename.find('/') == -1:
            filename = self._DEFAULT_DIRECTORY + filename
        config = configparser.ConfigParser()
        if not os.path.isfile(filename):
            raise Exception(filename + 'not found.')
        config.read(filename)

        self._config = config

    def get(self, section, value, isString = True):

        if self._config[section].get(section, value):
            value = self._config[section][value]
            return value if isString else eval(value)
        else:
            return None

    def set(self, section, value):

        if section in self._config:
            self._config[section] = value
        else:
            return None

    def eq(self, section, value, testvalue, isString = True):

        value = self.get(section, value, isString)
        return value == testvalue

    def getConfig(self):

        return self._config
