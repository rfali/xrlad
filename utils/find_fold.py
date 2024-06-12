import os
import pandas as pd

def find_fold(rid):
    """
    Finds and returns the fold number of a given record ID (rid) from a set of Excel files in a specified directory.

    Parameters:
    rid (int or str): The record ID to be searched for in the Excel files.

    Returns:
    str: The fold number of the record if found, otherwise None.

    Note:
    The function assumes that the Excel files are located in the 'dataset/adni' directory and that the fold number is the last character of the file name (excluding the extension).
    The function also assumes that the record IDs are stored in a column named 'RID' in a sheet named 'test' in the Excel files.
    """
        
    files = [file for file in os.listdir('dataset/adni') if not file.endswith("_parameters.xls")]

    for f in files:
        path = os.path.join('dataset/adni', f)

        df = pd.read_excel(path, sheet_name="test")

        if rid in df['RID'].values:
            return f[-5]    # extract n from file name 'adni_foldn.xls'

