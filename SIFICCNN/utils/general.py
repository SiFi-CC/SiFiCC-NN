import numpy as np
import os


def parent_directory(repo_name="SiFiCC-NN"):
    override = os.getenv("SIFICCNN_ROOT")
    if override:
        return os.path.abspath(override)

    # get current path, go two subdirectories higher
    # path = os.getcwd()
    path = os.getenv("PWD")
    while True:
        if os.path.basename(path) == repo_name:
            break
        parent = os.path.abspath(os.path.join(path, os.pardir))
        if parent == path:
            raise RuntimeError(
                f"Could not find repository root '{repo_name}'. Set SIFICCNN_ROOT to override it."
            )
        path = parent
    return path
