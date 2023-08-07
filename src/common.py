'''
Created on 20.05.2021

@author: Fernando Penaherrera @UOL/OFFIS
'''
import os
from os.path import join


def get_project_root():
    """
    Return the path to the project root directory.

    Returns:
        str: A directory path.
    """
    return os.path.realpath(os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    ))


def check_and_create_folders(folders):
    """
    Check if the specified folders exist; if not, create them.

    Args:
        folders (list[str]): A list of folder paths to check and create if missing.
    """
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder '{folder}' created.")
        else:
            pass
            # print(f"Folder '{folder}' already exists.")


BASE_DIR = get_project_root()
DATA_DIR = join(BASE_DIR, "data")
RESULTS_DIR = join(BASE_DIR, "results")
MODEL_RESSULTS_DIR = join(RESULTS_DIR, "model")
FIG_RESULTS_DIR = join(RESULTS_DIR, "figs")
SOURCE_DIR = join(BASE_DIR, "src")


def check_and_create_all_folders():
    """
    Check and create all the necessary project folders if missing.
    """
    folders_to_check = [
        BASE_DIR,
        DATA_DIR,
        RESULTS_DIR,
        MODEL_RESSULTS_DIR,
        FIG_RESULTS_DIR,
        SOURCE_DIR
    ]
    check_and_create_folders(folders_to_check)


if __name__ == '__main__':
    check_and_create_all_folders()
