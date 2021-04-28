"""
Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import pathlib

import numpy as np
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

np.random.seed(0)
import seaborn as sns

sns.set()
import pickle

from protein_attention.utils import get_reports_path, get_cache_path
from protein_attention.attention_analysis.background import binding_site_distribution,\
    contact_map_distribution, protein_modification_distribution
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_context("paper")
sns.set_style("white")


def to_filename(s, extension):
    return "".join(x if (x.isalnum() or x in "._-") else '_' for x in s) + "." + extension


def create_figure(feature_name, weighted_sum, weight_total, report_dir, min_total, filetype, max_seq_len,
                  use_bonferroni=False, k=10):
    assert filetype in ('png', 'pdf')

    mean_by_head = weighted_sum / weight_total
    n_layers, n_heads = mean_by_head.shape

    scored_heads = []
    for i in range(n_layers):
        for j in range(n_heads):
            if weight_total[i, j] > min_total:
                scored_heads.append((mean_by_head[i, j], i, j))
    top_heads = sorted(scored_heads, reverse=True)[:k]


    if feature_name == 'binding_site_to':
        counts = binding_site_distribution(max_seq_len)
        num_pos_background = counts[1]
        num_neg_background = counts[0]
    elif feature_name == 'protein_modification_to':
        counts = protein_modification_distribution(max_seq_len)
        num_pos_background = counts[1]
        num_neg_background = counts[0]
    elif feature_name == 'contact_map':
        counts = contact_map_distribution(max_seq_len)
        num_pos_background = counts[1]
        num_neg_background = counts[0]
    else:
        raise NotImplementedError
    num_total_background = num_pos_background + num_neg_background
    background_pct = num_pos_background / (num_total_background) * 100

    scores = []
    conf_ints = []
    labels = []

    for score, i, j in top_heads:
        scores.append(score.item() * 100)
        labels.append(f'{i + 1}-{j + 1}')
        num_pos = int(weighted_sum[i, j])
        num_total = int(weight_total[i, j])
        if use_bonferroni:
            print('m=', len(scored_heads))
            start, end = proportion_confint(num_pos, num_total, alpha=0.05 / len(scored_heads))
        else:
            start, end = proportion_confint(num_pos, num_total, alpha=0.05)
        conf_int = (end - start) / 2 * 100
        conf_ints.append(conf_int)
        print(i, j)
        print('background', num_pos_background, num_total_background, num_pos_background / num_total_background)
        print('attn', num_pos, num_total, num_pos / num_total, start, end)
        p_value = proportions_ztest([num_pos_background, num_pos], [num_total_background, num_total])
        print('p_value', f'{p_value[-1]:.25f}')

    print(list(enumerate(zip(labels, [f'{s:.3f}' for s in scores]))))

    figsize = (3.7, 1.8)
    plt.figure(figsize=figsize)
    plt.bar(range(k), scores, yerr=conf_ints, capsize=3, edgecolor="none")
    x = np.arange(k)
    plt.xticks(x, labels, fontsize=7, rotation=45)
    plt.xlabel(f'Top heads', fontsize=8, labelpad=6)
    plt.ylabel('Attention %', fontsize=8, labelpad=6)
    plt.yticks(fontsize=7)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', pad=-2)
    ax.axhline(background_pct, linestyle='dashed', color='#FF7F00', linewidth=2, alpha=0.8)
    plt.tight_layout()
    ax.grid(True, axis='y', color='#F3F2F3', lw=1.2)
    fname = report_dir / to_filename(feature_name, filetype)
    print('Saving', fname)
    plt.savefig(fname, format=filetype)
    plt.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', help='Name of experiment')
    parser.add_argument('--min_total', type=int, default=100)
    args = parser.parse_args()

    use_bonferroni = True
    if use_bonferroni:
        print('Using bonferroni')
    min_total = args.min_total
    filetype = 'pdf'
    print(args.exp_name)
    cache_path = get_cache_path() / f'{args.exp_name}.pickle'
    report_dir = get_reports_path() / 'attention_analysis' / f'{args.exp_name}_topheads'
    pathlib.Path(report_dir).mkdir(parents=True, exist_ok=True)

    cache_args, feature_to_weighted_sum, weight_total = pickle.load(open(cache_path, "rb"))
    with open(report_dir / 'args.json', 'w') as f:
        json.dump(vars(cache_args), f)
    for feature_name, weighted_sum in feature_to_weighted_sum.items():
        create_figure(feature_name, weighted_sum, weight_total, report_dir, min_total=min_total, filetype=filetype,
                      use_bonferroni=use_bonferroni, max_seq_len=cache_args.max_seq_len)
