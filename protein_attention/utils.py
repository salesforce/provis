"""Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def get_data_path() -> Path:
    """Returns data root folder."""
    return get_project_root() / Path('data')


def get_models_path() -> Path:
    """Returns models root folder."""
    return get_project_root() / Path('models')


def get_reports_path() -> Path:
    """Returns models root folder."""
    return get_project_root() / Path('reports')


def get_cache_path() -> Path:
    """Returns data root folder."""
    return get_data_path() / Path('cache')


### From https://github.com/songlab-cal/tape/blob/master/tape/utils/utils.py
import typing


# def write_lmdb(filename: str, iterable: typing.Iterable, map_size: int = 2 ** 20):
#     """Utility for writing a dataset to an LMDB file.
#     Args:
#         filename (str): Output filename to write to
#         iterable (Iterable): An iterable dataset to write to. Entries must be pickleable.
#         map_size (int, optional): Maximum allowable size of database in bytes. Required by LMDB.
#             You will likely have to increase this. Default: 1MB.
#     """
#     import lmdb
#     import pickle as pkl
#     env = lmdb.open(filename, map_size=map_size)
#
#     with env.begin(write=True) as txn:
#         for i, entry in enumerate(iterable):
#             txn.put(str(i).encode(), pkl.dumps(entry))
#         txn.put(b'num_examples', pkl.dumps(i + 1))
#     env.close()
