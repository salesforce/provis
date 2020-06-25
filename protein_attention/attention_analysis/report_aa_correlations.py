"""Report pairwise correlations of attention to specific amino acids, and compare to blosum

Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import pickle
import re

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from Bio.SubsMat.MatrixInfo import blosum62

from protein_attention.utils import get_reports_path, get_cache_path

sns.set()
np.random.seed(0)


def to_filename(s):
    return "".join(x if (x.isalnum() or x in "._-") else '_' for x in s)


def create_figures(feature_to_weighted_sums, weight_totals, min_total, report_dir, filetype):
    aa_blosum = set()
    for aa1, aa2 in blosum62.keys():
        aa_blosum.add(aa1)
        aa_blosum.add(aa2)

    include_mask = weight_totals >= min_total

    p = re.compile(r'aa_to_([A-Z])$')
    aa_to_features = {}
    for feature_name, weighted_sums in feature_to_weighted_sums.items():
        m = p.match(feature_name)
        if m:
            aa = m[1]
            mean_by_heads = np.where(include_mask, weighted_sums / weight_totals, -1)
            feature_vector = mean_by_heads.flatten()
            feature_vector = feature_vector[feature_vector != -1]
            aa_to_features[aa] = feature_vector

    aas = sorted(aa_to_features.keys())
    aas_set = set(aas)
    print('Excluding following AAs not in feature set', aa_blosum - aas_set)
    print('Excluding following AAs not in blosum62', aas_set - aa_blosum)
    aa_list = sorted(list(aas_set & aa_blosum))
    n_aa = len(aa_list)
    corr = np.zeros((n_aa, n_aa))
    for i, aa1 in enumerate(aa_list):
        vector1 = aa_to_features[aa1]
        for j, aa2 in enumerate(aa_list):
            if i == j:
                corr[i, j] = None
            else:
                vector2 = aa_to_features[aa2]
                corr[i, j], _ = pearsonr(vector1, vector2)

    cmap = 'Blues'
    ax = sns.heatmap(corr, cmap=cmap, vmin=-0.5)
    ax.set_xticklabels(aa_list)
    ax.set_yticklabels(aa_list)
    plt.savefig(report_dir / f'aa_corr_to.pdf', format=filetype)
    plt.close()

    blosum = np.zeros((n_aa, n_aa))
    for i, aa1 in enumerate(aa_list):
        for j, aa2 in enumerate(aa_list):
            if i == j:
                blosum[i, j] = None
            else:
                if blosum62.get((aa1, aa2)) is not None:
                    blosum[i, j] = blosum62.get((aa1, aa2))
                else:
                    blosum[i, j] = blosum62.get((aa2, aa1))

    ax = sns.heatmap(blosum, cmap=cmap, vmin=-4, vmax=4)
    ax.set_xticklabels(aa_list)
    ax.set_yticklabels(aa_list)
    plt.savefig(report_dir / f'blosum62.pdf',
                format=filetype)
    plt.close()

    corr_scores = []
    blos_scores = []
    for i in range(n_aa):
        for j in range(i):
            corr_scores.append(corr[i, j])
            blos_scores.append(blosum[i, j])
    print('Pearson Correlation between feature corr and blosum',
          pearsonr(corr_scores, blos_scores)[0])


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', help='Name of experiment')
    args = parser.parse_args()

    min_total = 100
    filetype = 'pdf'

    cache_path = get_cache_path() / f'{args.exp_name}.pickle'
    args, feature_to_weighted_sums, weight_totals = pickle.load(open(cache_path, "rb"))
    print(args)
    print(weight_totals)

    report_dir = get_reports_path() / 'attention_analysis/blosum' / args.exp_name
    report_dir.mkdir(parents=True, exist_ok=True)

    create_figures(feature_to_weighted_sums, weight_totals, min_total, report_dir, filetype)

    with open(report_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f)
