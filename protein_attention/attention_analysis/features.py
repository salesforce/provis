"""
Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from abc import ABC, abstractmethod


class EdgeFeature(ABC):

    @abstractmethod
    def get_values(self, item, from_index, to_index):
        pass


class SecStructFeature(EdgeFeature):

    def __init__(self, include_from=False, include_to=True):
        self.include_from = include_from
        self.include_to = include_to
        assert include_from or include_to

    def get_values(self, seq, from_index, to_index):
        feature_values = {}
        if self.include_from:
            from_secstruct = seq['secondary'][from_index]
            feature_name = f'sec_struct_from_{from_secstruct}'
            feature_values[feature_name] = 1
        if self.include_to:
            to_secstruct = seq['secondary'][to_index]
            feature_name = f'sec_struct_to_{to_secstruct}'
            feature_values[feature_name] = 1

        return feature_values


class AminoAcidFeature(EdgeFeature):

    def __init__(self, include_from=False, include_to=True):
        self.include_from = include_from
        self.include_to = include_to
        assert include_from or include_to

    def get_values(self, item, from_index, to_index):
        feature_values = {}
        if self.include_from:
            feature_name = f'aa_from_{item["primary"][from_index]}'
            feature_values[feature_name] = 1
        if self.include_to:
            feature_name = f'aa_to_{item["primary"][to_index]}'
            feature_values[feature_name] = 1
        return feature_values


class BindingSiteFeature(EdgeFeature):

    def get_values(self, item, from_index, to_index):
        feature_values = {}
        if item['site_indic'][to_index] == 1:
            feature_values = {
                'binding_site_to': 1
            }
        return feature_values


class ContactMapFeature(EdgeFeature):
    def get_values(self, item, from_index, to_index):
        contact_map = item['contact_map']
        contact1 = contact_map[from_index, to_index]
        contact2 = contact_map[to_index, from_index]
        assert contact1 == contact2
        if contact1 == 1:
            return {'contact_map': 1}
        else:
            return {}
#
#
# if __name__ == "__main__":
#     from protein_interpret.datasets import BindingSiteDataset
#     from protein_interpret.utils import get_data_path
#     from protein_interpret.attention_analysis.compute_edge_features import convert_item
#
#     model_name = 'bert'
#     d = BindingSiteDataset(get_data_path(), 'train')
#     ds_name = 'binding_sites'
#     idx = 0
#     data = d.data[idx]
#     x = d[idx]
#     item = convert_item(ds_name, x, data, model_name)
#     print(data['id'])
#
#     print(list(zip(range(0, len(item['primary'])), item['primary'], item['site_indic'])))
#     feature = BindingSiteFeature()
#
#     assert feature.get_values(item, 23, 88)['binding_site_to'] == 1
#     assert feature.get_values(item, 22, 88)['binding_site_to'] == 1
#     assert 'binding_site_to' not in feature.get_values(item, 23, 90)
#









