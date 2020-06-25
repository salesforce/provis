"""Create combined plot from multiple features

Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from protein_attention.attention_analysis.report_edge_features import contact_map_pattern, \
    secondary_to_pattern, binding_site_pattern
from protein_attention.utils import get_reports_path, get_cache_path

ss4_names = {
    '0': 'Helix',
    '1': 'Strand',
    '2': 'Turn/Bend'
}

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_names', nargs='+', help='Names of experiments')
    args = parser.parse_args()

    min_total = 100
    filetype = 'pdf'

    report_dir = get_reports_path() / 'attention_analysis/edge_features_combined'
    pathlib.Path(report_dir).mkdir(parents=True, exist_ok=True)

    feature_data = []
    include_features = [contact_map_pattern, secondary_to_pattern, binding_site_pattern]
    for exp in args.exp_names:
        cache_path = get_cache_path() / f'{exp}.pickle'
        args, feature_to_weighted_sum, weight_total = pickle.load(open(cache_path, "rb"))
        with open(report_dir / f'args_{exp}.json', 'w') as f:
            json.dump(vars(args), f)
        for feature, weighted_sum in feature_to_weighted_sum.items():
            for p in include_features:
                m = p.match(feature)
                desc = None
                if m:
                    if p == contact_map_pattern:
                        desc = 'Contact'
                    elif p == binding_site_pattern:
                        desc = 'Binding Site'
                    elif p == secondary_to_pattern:
                        sec = m.group(1)
                        desc = ss4_names.get(sec)
                    else:
                        raise ValueError
                    break
            if not desc:
                continue
            mean_by_head = weighted_sum / weight_total
            exclude_mask = np.array(weight_total) < min_total
            masked_mean_by_head = np.ma.masked_array(mean_by_head, mask=exclude_mask)
            layer_macro = masked_mean_by_head.mean(-1)
            layer_macro *= 100  # Convert to percentage
            n_layers = len(layer_macro)
            assert n_layers == 12
            normalized = layer_macro / layer_macro.sum()
            assert np.allclose(normalized.sum(), 1)
            mean_center = sum(i * normalized[i] for i in range(n_layers))
            feature_data.append((mean_center, feature, desc, layer_macro))

    # Sort aggregated data by center of gravity
    feature_data.sort()

    # Create combined plot
    figsize = (3, 5)
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(len(feature_data), figsize=figsize, sharex=True, gridspec_kw={'wspace': 0, 'hspace': .17})
    for i, (center, feature, desc, layer_macro) in enumerate(feature_data):
        ax[i].plot(list(range(12)),
                   layer_macro)
        ax[i].axvline(x=center, color='red', linestyle='dashed', linewidth=1)
        ax[i].tick_params(labelsize=6)
        ax[i].set_ylabel(desc, fontsize=8)
        ax[i].set_ylim(top=1.03 * max(layer_macro), bottom=0)
        ax[i].yaxis.tick_right()
        formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
        ax[i].yaxis.set_major_formatter(formatter)

    plt.xticks(range(12), range(1, 13))
    plt.xlabel('Layer', fontsize=8)
    fname = report_dir / f'combined_features.{filetype}'
    print('Saving', fname)
    plt.savefig(fname, format=filetype, bbox_inches='tight')
    plt.close()
