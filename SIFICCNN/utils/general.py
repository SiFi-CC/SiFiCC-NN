import numpy as np
import os


def parent_directory(repo_name="SiFiCC-NN"):
    # get current path, go two subdirectories higher
    path = os.getcwd()
    while True:
        if os.path.basename(path) == repo_name:
            break
        path = os.path.abspath(os.path.join(path, os.pardir))
    return path


