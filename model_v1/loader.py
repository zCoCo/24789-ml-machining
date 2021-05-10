"""
Methods for loading data from the .mat files.
"""
from typing import Dict, List, Tuple, Optional
import attr

import os

from data import MachiningDataset, ExperimentData, ChannelData
from logger import logger
from opts import get_opts

opts = get_opts()


def load_all_mats(data_dir: Optional[str] = None, default_key: str = 'root') -> Tuple[Dict[str, ExperimentData], MachiningDataset]:
    """
    Recursively loads all `*.mat` files in the given directory.

    If there are top-level subdirectories in the given directory, each directory 
    will be given a key in a dictionary. If the directory just contains `*.mat` 
    files, the data structures will be loaded into a key with the name given by 
    `default_key`.

    If no directory is given, `opts.data_dir` will be used.
    """
    logger.info("Loading all data . . .")
    if data_dir is None:
        data_dir = opts.data_dir

    data: Dict[str, List[ChannelData]] = {}

    def add_item_if_data(dir_key: str, item_dir: str, item_name: str, recursive=True):
        """
        Adds the given directory item to the data dictionary under the key 
        `dir_key` if it's data (a *.mat file).

        If recursive is set to `True` and the directory item is a directory, 
        that directory will be opened and its contents will be searched for data 
        files. Any data files found will be added under the same `dir_key` given 
        here. Any further directories encountered will also be recursed, etc.

        Args:
            dir_key (str): Key this data item should be added under in the `data` dict.
            item_dir (str): Directory where this directory item was found.
            item_name (str): Name of this directory item (i.e. what's inside what's returned from `os.listdir`)
        """
        *name_parts, ext = item_name.split('.')
        name = '.'.join(name_parts)
        if ext == 'mat':
            data[dir_key].append(ChannelData.from_mat(item_dir, name, ext))

        item_path = os.path.join(item_dir, item_name)
        if recursive and os.path.isdir(item_path):
            for sub_item_name in os.listdir(item_path):
                add_item_if_data(
                    dir_key=dir_key,
                    item_dir=item_path,
                    item_name=sub_item_name,
                    recursive=True
                )

    for item_name in os.listdir(data_dir):
        # Grab all .mat files in root:
        data[default_key] = []
        add_item_if_data(
            dir_key=default_key,
            item_dir=data_dir,
            item_name=item_name,
            recursive=False
        )

        # If item is a (top-level) subdirectory, add it to dict and parse all its contents:
        item_path = os.path.join(data_dir, item_name)
        if os.path.isdir(item_path):
            data[item_name] = []
            for sub_item_name in os.listdir(item_path):
                add_item_if_data(
                    dir_key=item_name,
                    item_dir=item_path,
                    item_name=sub_item_name,
                    recursive=True
                )

    out_data: Dict[str, ExperimentData] = {}
    # Construct `ExperimentData` collections from collected data:
    for k, c in data.items():
        # Only pass through non-empty collections:
        if len(c) > 0:
            # Process data:
            out_data[k] = ExperimentData.from_mat(dir_name=k, data=c)

    # Create torch dataset from all experiment data:
    dataset = MachiningDataset([*out_data.values()])

    logger.notice("All data loaded.")  # type: ignore
    return out_data, dataset
