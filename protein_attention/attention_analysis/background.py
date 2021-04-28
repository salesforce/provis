from collections import Counter

import tqdm
from tape.datasets import ProteinnetDataset

from protein_attention.datasets import BindingSiteDataset, ProteinModificationDataset
from protein_attention.utils import get_data_path


def binding_site_distribution(max_len):
    d = BindingSiteDataset(get_data_path(), 'train')
    c = Counter()
    for row in tqdm.tqdm(d):
        site_indic = row[-1]
        c.update(site_indic[:max_len])

    return c


def protein_modification_distribution(max_len):
    d = ProteinModificationDataset(get_data_path(), 'train')
    c = Counter()
    for row in tqdm.tqdm(d):
        mod_indic = row[-1]
        c.update(mod_indic[:max_len])
    return c


def contact_map_distribution(max_len):
    d = ProteinnetDataset(get_data_path(), 'train')
    c = Counter()
    for row in tqdm.tqdm(d):
        contact_map = row[-2][:max_len, :max_len].flatten()
        c.update(contact_map)
    return c
