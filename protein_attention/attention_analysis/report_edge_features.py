"""
Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import pathlib

import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set()
import pickle

from protein_attention.utils import get_reports_path, get_cache_path
from matplotlib.ticker import FuncFormatter

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import re

secondary_names = {
    '0': 'Helix',
    '1': 'Beta',
    '2': 'Turn/Bend',
    '3': 'Blank'
}

aa_to_pattern = re.compile(r'res_to_([A-Z])$')
secondary_to_pattern = re.compile(r'sec_struct_to_([A-Z0-3\s])$')
contact_map_pattern = re.compile(r'contact_map')
binding_site_pattern = re.compile(r'binding_site_to')


def to_filename(s, extension):
    return "".join(x if (x.isalnum() or x in "._-") else '_' for x in s) + "." + extension


def create_figure(feature_name, weighted_sum, weight_total, report_dir, min_total, filetype):
    assert filetype in ('png', 'pdf')

    mean_by_head = weighted_sum / weight_total
    exclude_mask = np.array(weight_total) < min_total

    masked_mean_by_head = np.ma.masked_array(mean_by_head, mask=exclude_mask)
    layer_macro = masked_mean_by_head.mean(-1)

    plt.figure(figsize=(3, 2.2))
    ax1 = plt.subplot2grid((100, 85), (0, 0), colspan=65, rowspan=99)
    ax2 = plt.subplot2grid((100, 85), (12, 70), colspan=15, rowspan=75)
    heatmap = sns.heatmap((mean_by_head * 100).tolist(), center=0.0, ax=ax1,
                          square=True, cbar=False, linewidth=0.1, linecolor='#D0D0D0',
                          cmap=LinearSegmentedColormap.from_list('rg', ["#F14100", "white", "#3ED134"], N=256),
                          mask=exclude_mask,
                          xticklabels=['', '2', '', '4', '', '6', '', '8', '', '10', '', '12'],
                          yticklabels=['', '2', '', '4', '', '6', '', '8', '', '10', '', '12'])

    for _, spine in heatmap.spines.items():
        spine.set_visible(True)
    plt.setp(heatmap.get_yticklabels(), fontsize=7)
    plt.setp(heatmap.get_xticklabels(), fontsize=7)
    heatmap.tick_params(axis='x', pad=1, length=2)
    heatmap.tick_params(axis='y', pad=1, length=2)
    heatmap.yaxis.labelpad = 2
    heatmap.invert_yaxis()
    heatmap.set_facecolor('#E7E6E6')
    # split axes of heatmap to put colorbar
    ax_divider = make_axes_locatable(ax1)
    cax = ax_divider.append_axes('left', size='7%', pad='33%')
    # # make colorbar for heatmap.
    # # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    cbar = colorbar(ax1.get_children()[0], cax=cax, orientation='vertical', format='%.0f%%')
    cax.yaxis.set_ticks_position('left')
    cbar.solids.set_edgecolor("face")
    cbar.ax.tick_params(labelsize=7, length=4, pad=2)
    ax1.set_title('% Attention', size=9)
    ax1.set_xlabel('Head', size=8)
    ax1.set_ylabel('Layer', size=8)
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
    ax2.set_title('      Layer Avg.', size=9)
    # bp = sns.barplot(x=layer_macro * 100, ax=ax2, y=list(range(layer_macro.shape[0])), color="#3D4FC4", orient="h")
    bp = sns.barplot(x=layer_macro * 100, ax=ax2, y=list(range(layer_macro.shape[0])), color="#556FAB", orient="h")
    formatter = FuncFormatter(lambda y, pos: '0' if (y == 0) else "%d%%" % (y))
    ax2.xaxis.set_major_formatter(formatter)
    plt.setp(bp.get_xticklabels(), fontsize=7)
    bp.tick_params(axis='x', pad=1, length=3)
    ax2.invert_yaxis()
    ax2.set_yticklabels([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.axvline(0, linewidth=.85, color='black')
    fname = report_dir / to_filename(feature_name, filetype)
    print('Saving', fname)
    plt.savefig(fname, format=filetype)
    plt.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', help='Name of experiment')
    args = parser.parse_args()
    print(args)
    min_total = 100
    filetype = 'pdf'

    cache_path = get_cache_path() / f'{args.exp_name}.pickle'
    report_dir = get_reports_path() / 'attention_analysis' / args.exp_name
    pathlib.Path(report_dir).mkdir(parents=True, exist_ok=True)

    args, feature_to_weighted_sum, weight_total = pickle.load(open(cache_path, "rb"))
    with open(report_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f)
    print(args)
    for feature_name, weighted_sum in feature_to_weighted_sum.items():
        create_figure(feature_name, weighted_sum, weight_total, report_dir, min_total=min_total, filetype=filetype)
