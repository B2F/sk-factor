import configparser
import os
import tomllib

class Config():

    _DEFAULT_DIRECTORY = 'config/'
    _config: dict

    def __init__(self, filename = 'sk_factor.ini'):

        if filename.find('/') == -1:
            filename = self._DEFAULT_DIRECTORY + filename + '.toml'
        config = configparser.ConfigParser()
        if not os.path.isfile(filename):
            raise Exception(filename + 'not found.')

        with open(filename, "rb") as f:
            config = tomllib.load(f)

        self._config = config

    def get(self, section, value):

        if self._config[section].get(section, value):
            value = self._config[section][value]
            return value
        else:
            return None

    def set(self, section, value):

        if section in self._config:
            self._config[section] = value
        else:
            return None

    def eq(self, section, value, testvalue):

        value = self.get(section, value)
        return value == testvalue

    def getConfig(self):

        return self._config
