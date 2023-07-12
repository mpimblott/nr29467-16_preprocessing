from pathlib import Path

from preprocessing.files.hdf5 import H5Handler
from preprocessing.files.npy import NPYHandler
from preprocessing.files.tif import TiffHandler

def get_handler(path: Path):
    for handler in [H5Handler, NPYHandler, TiffHandler]:
        if handler.check_ext(path):
            return handler
    raise ValueError(f'No handler found for {path}')