import os
import pandas as pd
import re
import time

class Files():

    @staticmethod
    def toCsv(df: pd.DataFrame, directory: str, filename: str, hasTimestamp: bool = False):

        filename = re.match(r'(?:(?:(?:[^\/]*)\/)*)([^\.]*)(?:\..*)?$', filename).group(1)

        targetDirectory = re.match(r'^(.*\/)+', f"{directory}/{filename}").group(0)
        filename = re.match(r'(?:(?:(?:[^\/]*)\/)*)(.*)$', f"{directory}/{filename}").group(1)

        if not os.path.isdir(targetDirectory):
            os.makedirs(targetDirectory)

        if hasTimestamp:
            filename += '_' + str(time.time())

        fullPath = f"{targetDirectory}{filename}.csv"
        df.to_csv(f"{fullPath}")

        print (f'{fullPath} written to disk.')
