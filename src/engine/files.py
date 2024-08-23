import os
import pandas as pd
import re
import time

class Files():

    @staticmethod
    def toCsv(df: pd.DataFrame, directory: str, filename: str = None, hasTimestamp: bool = False, index = True):

        # If filename is None then directory contains full path to file.
        if filename:
            directory = f"{directory}/{filename}"

        targetDirectory = re.match(r'^(.*\/)+', f"{directory}").group(0)
        # Removes path and extension from models config list / argument:
        filename = re.match(r'(?:(?:(?:[^\/]*)\/)*)([^\.]*)(?:\..*)?$', f"{directory}").group(1)

        if not os.path.isdir(targetDirectory):
            os.makedirs(targetDirectory)

        if hasTimestamp:
            filename += '_' + str(time.time())

        fullPath = f"{targetDirectory}{filename}.csv"
        df.to_csv(f"{fullPath}", index=index)

        print (f'{fullPath} written to disk.')
