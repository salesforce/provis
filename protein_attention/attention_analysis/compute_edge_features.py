"""Compute aggregate statistics of attention edge features over a dataset

Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import defaultdict

import torch
from tqdm import tqdm

from protein_attention.attention_analysis.utils import AttentionHelper


def analyze_heads(model,
                  n_layers,
                  n_heads,
                  items,
                  features,
                  start_index,
                  tokenizer,
                  model_name,
                  cuda=True,
                  max_seq_len=None,
                  min_attn=None):

    model.eval()

    with torch.no_grad():

        # Dictionary that maps feature_name to array of shape (n_layers, n_heads), containing
        # weighted sum of feature values for each layer/head over all examples
        feature_to_weighted_sum = defaultdict(lambda: torch.zeros((n_layers, n_heads), dtype=torch.double))

        # Sum of attention_analysis weights in each layer/head over all examples
        weight_total = torch.zeros((n_layers, n_heads), dtype=torch.double)

        for item in tqdm(items):
            if model_name == 'bert':
                tokens = item['primary']
                if max_seq_len:
                    tokens = tokens[:max_seq_len - 2]  # Account for SEP, CLS tokens (added in next step)
                token_idxs = tokenizer.encode(tokens).tolist()
                if max_seq_len:
                    assert len(token_idxs) == min(len(tokens) + 2, max_seq_len)
                else:
                    assert len(token_idxs) == len(tokens) + 2
            else:
                raise ValueError

            inputs = torch.tensor(token_idxs).unsqueeze(0)
            if cuda:
                inputs = inputs.cuda()
            attns = model(inputs)[-1]

            if model_name == 'bert':
                # Remove attention from <CLS> (first) and <SEP> (last) token
                attns = [attn[:, :, 1:-1, 1:-1] for attn in attns]

            if 'contact_map' in item:
                assert (item['contact_map'].shape == attns[0][0, 0].shape) or (attns[0][0, 0].shape[0] == max_seq_len - 2), \
                    (item['id'], item['contact_map'].shape, attns[0][0, 0].shape)
            if 'site_indic' in item:
                assert (item['site_indic'].shape == attns[0][0, 0, 0].shape) or (attns[0][0, 0].shape[0] == max_seq_len - 2),\
                    item['id']

            if min_attn is not None:
                # Set attention weights below min_attn to zero
                attns = [(attn > min_attn) * attn for attn in attns]

            ah = AttentionHelper(attns, cpu=True)
            assert ah.n_heads == n_heads
            assert ah.n_layers == n_layers

            # Update total attention_analysis weights per head. Sum over from_index (dim 2), to_index (dim 3)
            weight_total += ah.attns[:, :, start_index:, start_index:].sum((2, 3))

            # Update weighted sum of feature values per head
            for to_index in range(start_index, ah.seq_len):
                for from_index in range(start_index, ah.seq_len):
                    for feature in features:
                        # Compute feature values
                        feature_dict = feature.get_values(item, from_index, to_index)
                        for feature_name, value in feature_dict.items():
                            # Update weighted sum of feature values across layers and heads
                            feature_to_weighted_sum[feature_name] += ah.attns[:, :, from_index, to_index] * value

        if (weight_total == 0).any():
            print('WARNING: weight totals are zero in some heads')
        return feature_to_weighted_sum, weight_total


def convert_item(dataset_name, x, data, model_name, features):

    item = {}
    try:
        item['id'] = data['id']
    except ValueError:
        item['id'] = data['id'].decode('utf8')

    item['primary'] = data['primary']
    if dataset_name == 'proteinnet':
        if 'contact_map' in features:
            token_ids, input_mask, contact_map, protein_length = x
            item['contact_map'] = contact_map
    elif dataset_name == 'secondary':
        if 'ss4' in features:
            ss8_blank_index = 7
            ss4_blank_index = 3
            item['secondary'] = [ss4_blank_index if ss8 == ss8_blank_index else ss3 for ss3, ss8 in \
                                    zip(data['ss3'], data['ss8'])]
    elif dataset_name == 'binding_sites':
        if 'binding_sites' in features:
            token_ids, input_mask, site_indic = x
            item['site_indic'] = site_indic
    else:
        raise ValueError

    if model_name == 'bert':
        # Remove attention from <CLS> (first) and <SEP> (last) token
        if 'site_indic' in item:
            item['site_indic'] = item['site_indic'][1:-1]
    else:
        raise NotImplementedError  # SEP/CLS Padding already included

    return item


if __name__ == "__main__":
    import pickle
    import pathlib

    from tape import TAPETokenizer, ProteinBertModel
    from tape.datasets import ProteinnetDataset, SecondaryStructureDataset

    from protein_attention.datasets import BindingSiteDataset
    from protein_attention.utils import get_cache_path, get_data_path
    from protein_attention.attention_analysis.features import AminoAcidFeature, SecStructFeature, BindingSiteFeature, \
        ContactMapFeature

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', required=True, help='Name of experiment. Used to create unique filename.')
    parser.add_argument('--features', nargs='+', required=True, help='list of features')
    parser.add_argument('--dataset', required=True, help='Dataset id')
    parser.add_argument('--num-sequences', type=int, required=True, help='Number of sequences to analyze')
    parser.add_argument('--model', default='bert', help='Name of model.')
    parser.add_argument('--model_dir', help='Optional directory where pretrained model is located')
    parser.add_argument('--shuffle', action='store_true', help='Whether to randomly shuffle data')
    parser.add_argument('--max-seq-len', type=int, required=True, help='Max sequence length')
    parser.add_argument('--start-index', type=int, default=0, help='Starting amino acid index to include in analysis')
    parser.add_argument('--seed', type=int, default=123, help='PyTorch seed')
    parser.add_argument('--min-attn', type=float, help='min attention value for inclusion in analysis')
    parser.add_argument('--no_cuda', action='store_true', help='CPU only')
    args = parser.parse_args()
    print(args)

    if args.num_sequences is not None and not args.shuffle:
        print('WARNING: You are using a subset of sequences and you are not shuffling the data. This may result '
              'in a skewed sample.')
    cuda = not args.no_cuda

    torch.manual_seed(args.seed)

    if args.dataset == 'proteinnet':
        dataset = ProteinnetDataset(get_data_path(), 'train')
    elif args.dataset == 'secondary':
        dataset = SecondaryStructureDataset(get_data_path(), 'train')
    elif args.dataset == 'binding_sites':
        dataset = BindingSiteDataset(get_data_path(), 'train')
    else:
        raise ValueError(f"Invalid dataset id: {args.dataset}")

    if not args.num_sequences:
        raise NotImplementedError

    if args.model == 'bert':
        tokenizer = TAPETokenizer()
        if args.model_dir:
            model_version = args.model_dir
        else:
            model_version = 'bert-base'
        model = ProteinBertModel.from_pretrained(model_version, output_attentions=True)
        if cuda:
            model.to('cuda')
        num_layers = 12
        num_heads = 12
        unidirectional = False
    else:
        raise ValueError(f"Invalid model: {args.model}")

    if args.shuffle:
        random_indices = torch.randperm(len(dataset))[:args.num_sequences].tolist()
        items = []
        print('Loading dataset')
        for i in tqdm(random_indices):
            item = convert_item(args.dataset, dataset[i], dataset.data[i], args.model, args.features)
            items.append(item)
    else:
        raise NotImplementedError

    features = []
    for feature_name in args.features:
        if feature_name == 'aa':
            features.append(AminoAcidFeature())
        elif feature_name == 'ss4':
            features.append(SecStructFeature())
        elif feature_name == 'binding_sites':
            features.append(BindingSiteFeature())
        elif feature_name == 'contact_map':
            features.append(ContactMapFeature())
        else:
            raise ValueError(f"Invalid feature name: {feature_name}")

    feature_to_weighted_sum, weight_total = analyze_heads(
        model,
        num_layers,
        num_heads,
        items,
        features,
        args.start_index,
        tokenizer,
        args.model,
        cuda,
        max_seq_len=args.max_seq_len,
        min_attn=args.min_attn)


    cache_dir = get_cache_path()
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{args.exp_name}.pickle'
    pickle.dump((args, dict(feature_to_weighted_sum), weight_total), open(path, 'wb'))
    print('Wrote to', path)