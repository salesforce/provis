"""Extensions to dataset classes from TAPE Repository: https://github.com/songlab-cal/tape
Date         Change
----------   ---------------------
05/01/2020   Added Binding Site dataset class
             Added one-vs-all Secondary structure dataset class

"""

from pathlib import Path
from typing import Union, List, Tuple, Sequence, Dict, Any

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from tape.datasets import dataset_factory
from tape.tokenizers import TAPETokenizer
from torch.utils.data import Dataset

ss8_cds = ['G', 'H', 'I', 'B', 'E', 'S', 'T', ' ']
ss8_to_idx = {cd: i for i, cd in enumerate(ss8_cds)}

ss8_blank_index = 7
ss4_blank_index = 3


class SecondaryStructureOneVsAllDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 label_scheme: str,
                 label: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):

        if label_scheme != 'ss8' and label_scheme != 'ss4':
            raise NotImplementedError

        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        if label_scheme == 'ss8':
            self.label = ss8_to_idx[label]
        elif label_scheme == 'ss4':
            self.label = label
        else:
            raise NotImplementedError
        self.label_scheme = label_scheme

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        if self.label_scheme == 'ss4':
            # ss8 code 7 is for blank label. 3 is used to represent blank in ss4
            ss_labels = [ss4_blank_index if ss8 == ss8_blank_index else ss3 for ss3, ss8 in
                         zip(item['ss3'], item['ss8'])]
        else:
            ss_labels = item['ss8']
        labels = np.asarray([label == self.label for label in ss_labels], np.int64)
        # pad with -1s because of cls/sep tokens
        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, -1))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


class BindingSiteDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False,
                 max_seqlen: int = 512):

        allowed_splits = ('train', 'valid')
        if split not in allowed_splits:
            raise ValueError(f"Unrecognized split: {split}. Must be one of: {', '.join(allowed_splits)}")

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'binding_sites/binding_site_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        self.max_seqlen = max_seqlen

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        sequence = item['primary']
        positions = item['positions']
        if self.max_seqlen:
            sequence = sequence[:self.max_seqlen]
            positions = positions[:self.max_seqlen]

        token_ids = self.tokenizer.encode(sequence)
        input_mask = np.ones_like(token_ids)

        labels = [1 if seq_pos in item['sites'] else 0 for seq_pos in positions]

        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        label = torch.from_numpy(pad_sequences(label, -1))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': label}

        return output


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array
