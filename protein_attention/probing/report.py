"""Report on diagnostic classifiers for probing analysis

Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from protein_attention.utils import get_data_path
from protein_attention.utils import get_reports_path

sns.set()

ss4_names = {
    0: 'Helix',
    1: 'Strand',
    2: 'Turn/Bend'
}

feature_order = [
    'Helix',
    'Turn/Bend',
    'Strand',
    'Binding Site',
    'Contact Map'
]

feature_to_metric = {
    'Helix': 'F1',
    'Turn/Bend': 'F1',
    'Strand': 'F1',
    'Binding Site': 'Precision @ L/20',
    'Contact Map': 'Precision @ L/5'
}

feature_to_title = {
    'Helix': 'Secondary Structure: Helix',
    'Turn/Bend': 'Secondary Structure: Turn/Bend',
    'Strand': 'Secondary Structure: Strand',
    'Binding Site': 'Binding Site',
    'Contact Map': 'Contact Map'
}


def report(features, feature_scores, report_dir, filetype='pdf'):

    # Create detail plots
    feature_to_scores = dict(zip(features, feature_scores))
    for i, feature in enumerate(feature_order):
        scores = feature_to_scores[feature]
        fig, ax = plt.subplots()
        ax.plot(list(range(12)), scores)
        ax.set_xlabel('Layer', labelpad=10, fontsize=13)
        ax.set_title(feature_to_title[feature], pad=12, fontsize=13)
        ax.set_ylabel(feature_to_metric.get(feature, ''), labelpad=10, fontsize=13)
        fname = report_dir / f'layer_probing_{feature.replace(" ", "_").replace("/", "")}.{filetype}'
        print('Saving', fname)
        plt.xticks(range(12), range(1, 13))
        plt.tight_layout()
        plt.savefig(fname, format=filetype)  # , bbox_inches='tight')
        plt.close()
        scores = np.array(scores)
        if scores.sum() > 0:
            normalized = scores / scores.sum()
            assert np.allclose(normalized.sum(), 1)
            mean_center = sum(i * normalized[i] for i in range(12))
            print(feature, 'center:', mean_center)

    # Create combined plot of layer differences
    figsize = (3, 5)
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(len(feature_order), figsize=figsize, sharex=True,
                           gridspec_kw={'wspace': 0, 'hspace': .17})
    for i, feature in enumerate(feature_order):
        scores = feature_to_scores[feature]
        diffs = [scores[i] - scores[i - 1] for i in range(1, 12)]
        ax[i].bar(list(range(11)), diffs)
        ax[i].tick_params(labelsize=6)
        ax[i].set_ylabel(feature.replace('Contact Map', 'Contact'), fontsize=8)
        ax[i].yaxis.tick_right()
    plt.xticks(list(range(11)), list(range(2, 13)))
    plt.xlabel('Layer', fontsize=8)
    fname = report_dir / f'multichart_layer_delta_probing.{filetype}'
    print('Saving', fname)
    plt.savefig(fname, format=filetype, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    data_path = get_data_path()

    features = []
    feature_scores = []

    # Probing sec struct results
    ss_cds = [0, 1, 2]
    ss_names = ss4_names
    for ss_cd in ss_cds:
        feature = ss_names[ss_cd]
        features.append(feature)
        scores = [0] * 12
        for num_layers in list(range(1, 13)):
            fname = data_path / 'probing' / f'secondary_{ss_cd}_{num_layers}/results.json'
            try:
                with open(fname) as infile:
                    results = json.load(infile)
                    f1 = results['f1']
                    print(feature, num_layers, f1)
                    scores[num_layers - 1] = f1
            except FileNotFoundError:
                print('Skipping', fname)
                continue
        feature_scores.append(scores)

    # Probing binding site results
    features.append('Binding Site')
    scores = [0] * 12
    for num_layers in list(range(1, 13)):
        fname = data_path / 'probing' / f'binding_sites_{num_layers}/results.json'
        try:
            with open(fname) as infile:
                results = json.load(infile)
                print('binding sites', num_layers, 'f1:', results['f1'], 'precision:', results['precision'], 'recall:',
                      results['recall'], 'precision at k:', results['precision_at_k'])
                scores[num_layers - 1] = results['precision_at_k']
        except FileNotFoundError:
            print('Skipping', fname)
            continue
    feature_scores.append(scores)

    # Probing contact map results
    features.append('Contact Map')
    scores = [0] * 12
    for num_layers in list(range(1, 13)):
        fname = data_path / 'probing' / f'contact_map_{num_layers}/results.json'
        try:
            with open(fname) as infile:
                results = json.load(infile)
                print('contact maps', num_layers, 'f1:', results['f1'], 'precision:', results['precision'], 'recall:',
                      results['recall'], 'precision at k:', results['precision_at_k'])
                scores[num_layers - 1] = results['precision_at_k']
        except FileNotFoundError:
            print('Skipping', fname)
            continue
    feature_scores.append(scores)

    report_dir = get_reports_path() / 'probing'
    pathlib.Path(report_dir).mkdir(parents=True, exist_ok=True)
    report(features, feature_scores, report_dir)
