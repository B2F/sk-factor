from preprocess.base_preprocessor import BasePreprocessor

class DropNa(BasePreprocessor):

    def preprocess(self, df):
      df.dropna(axis=0, inplace=True)
      df.reset_index(inplace=True)
      return df
