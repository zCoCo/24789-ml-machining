"""
Methods for loading data from the .mat files.
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import os
import scipy  # type: ignore

from opts import get_opts
opts = get_opts()


@dataclass(frozen=True, order=True)
class DataFile:
    name: str
    ext: int
    data: float = 0.0


def load_all_mats(data_dir: Optional[str], default_key: str = 'root') -> Dict[str, List[DataFile]]:
    """
    Recursively loads all `*.mat` files in the given directory.

    If there are top-level subdirectories in the given directory, each directory 
    will be given a key in a dictionary. If the directory just contains `*.mat` 
    files, the data structures will be loaded into a key with the name given by 
    `default_key`.

    If no directory is given, `opts.data_dir` will be used.
    """
    if data_dir is None:
        data_dir = opts.data_dir

    data: Dict[str, List[DataFile]] = {}

    # Grab all files in root:
    data[default_key] = []
    for file_name in os.listdir(data_dir):
        *name_parts, ext = file_name.split('.')
        name = '.'.join(name_parts)
        if ext == 'mat':
            data[default_key] = DataFile(name, ext)

    return data


def load_from_mat(addr: str) -> None:
    pass
