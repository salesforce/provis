"""Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch


class AttentionHelper:

    def __init__(self, attns, cpu=False):
        if attns[0].size(0) != 1:
            raise NotImplementedError(f"Currently only support batch size of 1, not {attns[0].size(0)}")
        self.attns = torch.stack([attn.squeeze(0) for attn in attns])
        if cpu:
            self.attns = self.attns.cpu()

    @property
    def n_layers(self):
        return self.attns.size(0)

    @property
    def n_heads(self):
        return self.attns.size(1)

    @property
    def seq_len(self):
        return self.attns.size(2)


if __name__ == "__main__":
    from protein_interpret.progen import load_progen, ProgenTokenizer
    import math

    model = load_progen()
    model.eval()
    seq = 'YMIQEEEWDRDL'
    tokenizer = ProgenTokenizer()
    tokens = tokenizer.encode(seq)
    input = torch.tensor(tokens).unsqueeze(0).cuda()
    with torch.no_grad():
        attns = model(input)[1]
    ah = AttentionHelper(attns)
    assert ah.n_layers == 36
    assert ah.n_heads == 16
    assert ah.seq_len == 12
    for layer in range(ah.n_layers):
        for head in range(ah.n_heads):
            for from_index in range(ah.seq_len):
                edge_sum = 0
                for to_index in range(ah.seq_len):
                    edge_weight = ah.attns[layer, head, from_index, to_index]
                    if to_index > from_index:
                        assert math.isclose(edge_weight, 0)
                    edge_sum += edge_weight
                assert math.isclose(edge_sum, 1, rel_tol=1e-5)

class MeanAccumulator:

    def __init__(self):
        self._weighted_sum = 0
        self._sum_weights = 0

    def add(self, x, weight=1):
        self._weighted_sum += x * weight
        self._sum_weights += weight

    def compute(self):
        return self._weighted_sum / self._sum_weights

