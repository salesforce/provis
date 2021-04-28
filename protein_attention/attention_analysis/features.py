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

    def get_values(self, item, from_index, to_index, dense=False):
        if item['site_indic'][to_index] == 1:
            return {'binding_site_to': 1}
        else:
            if dense:
                return {'binding_site_to': 0}
            else:
                return {}


class ProteinModificationFeature(EdgeFeature):

    def get_values(self, item, from_index, to_index, dense=False):
        if item['modification_indic'][to_index] == 1:
            return {'protein_modification_to': 1}
        else:
            if dense:
                return {'protein_modification_to': 0}
            else:
                return {}


class ContactMapFeature(EdgeFeature):
    def get_values(self, item, from_index, to_index, dense=False):
        contact_map = item['contact_map']
        contact1 = contact_map[from_index, to_index]
        contact2 = contact_map[to_index, from_index]
        assert contact1 == contact2
        if contact1 == 1:
            return {'contact_map': 1}
        else:
            if dense:
                return {'contact_map': 0}
            else:
                return {}
