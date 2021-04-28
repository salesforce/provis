"""Compute aggregate statistics of attention edge features over a dataset

Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import re
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


def compute_mean_attention(model,
                           n_layers,
                           n_heads,
                           items,
                           features,
                           tokenizer,
                           model_name,
                           model_version,
                           cuda=True,
                           max_seq_len=None,
                           min_attn=0):
    model.eval()

    with torch.no_grad():

        # Dictionary that maps feature_name to array of shape (n_layers, n_heads), containing
        # weighted sum of feature values for each layer/head over all examples
        feature_to_weighted_sum = defaultdict(lambda: torch.zeros((n_layers, n_heads), dtype=torch.double))

        # Sum of attention_analysis weights in each layer/head over all examples
        weight_total = torch.zeros((n_layers, n_heads), dtype=torch.double)

        for item in tqdm(items):
            # Get attention weights, shape is (num_layers, num_heads, seq_len, seq_len)
            attns = get_attention(model,
                                  item,
                                  tokenizer,
                                  model_name,
                                  model_version,
                                  cuda,
                                  max_seq_len)
            if attns is None:
                print('Skipping due to not returning attention')
                continue

            # Update total attention_analysis weights per head. Sum over from_index (dim 2), to_index (dim 3)
            mask = attns >= min_attn
            weight_total += mask.long().sum((2, 3))

            # Update weighted sum of feature values per head
            seq_len = attns.size(2)
            for to_index in range(seq_len):
                for from_index in range(seq_len):
                    for feature in features:
                        # Compute feature values
                        feature_dict = feature.get_values(item, from_index, to_index)
                        for feature_name, value in feature_dict.items():
                            # Update weighted sum of feature values across layers and heads
                            mask = attns[:, :, from_index, to_index] >= min_attn
                            feature_to_weighted_sum[feature_name] += mask * value

        return feature_to_weighted_sum, weight_total


def get_attention(model,
                  item,
                  tokenizer,
                  model_name,
                  model_version,
                  cuda,
                  max_seq_len):
    tokens = item['primary']
    if model_name == 'bert':
        if max_seq_len:
            tokens = tokens[:max_seq_len - 2]  # Account for SEP, CLS tokens (added in next step)
        if model_version in ('prot_bert', 'prot_bert_bfd', 'prot_albert'):
            formatted_tokens = ' '.join(list(tokens))
            formatted_tokens = re.sub(r"[UZOB]", "X", formatted_tokens)
            token_idxs = tokenizer.encode(formatted_tokens)
        else:
            token_idxs = tokenizer.encode(tokens)
        if isinstance(token_idxs, np.ndarray):
            token_idxs = token_idxs.tolist()
        if max_seq_len:
            assert len(token_idxs) == min(len(tokens) + 2, max_seq_len), (tokens, token_idxs, max_seq_len)
        else:
            assert len(token_idxs) == len(tokens) + 2
    elif model_name == 'xlnet':
        if max_seq_len:
            tokens = tokens[:max_seq_len - 2]  # Account for SEP, CLS tokens (added in next step)
        formatted_tokens = ' '.join(list(tokens))
        formatted_tokens = re.sub(r"[UZOB]", "X", formatted_tokens)
        token_idxs = tokenizer.encode(formatted_tokens)
        if isinstance(token_idxs, np.ndarray):
            token_idxs = token_idxs.tolist()
        if max_seq_len:
            # Skip rare sequence with this issue
            if len(token_idxs) != min(len(tokens) + 2, max_seq_len):
                print('Warning: the length of the sequence changed through tokenization, skipping')
                return None
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
    elif model_name == 'xlnet':
        # Remove attention from <CLS> (last) and <SEP> (second to last) token
        attns = [attn[:, :, :-2, :-2] for attn in attns]
    else:
        raise NotImplementedError

    if 'contact_map' in item:
        assert (item['contact_map'].shape == attns[0][0, 0].shape) or (attns[0][0, 0].shape[0] == max_seq_len - 2), \
            (item['id'], item['contact_map'].shape, attns[0][0, 0].shape)
    if 'site_indic' in item:
        assert (item['site_indic'].shape == attns[0][0, 0, 0].shape) or (attns[0][0, 0].shape[0] == max_seq_len - 2), \
            item['id']
    if 'modification_indic' in item:
        assert (item['modification_indic'].shape == attns[0][0, 0, 0].shape) or (
                    attns[0][0, 0].shape[0] == max_seq_len - 2), \
            item['id']

    attns = torch.stack([attn.squeeze(0) for attn in attns])
    return attns.cpu()


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
    elif dataset_name == 'protein_modifications':
        if 'protein_modifications' in features:
            token_ids, input_mask, modification_indic = x
            item['modification_indic'] = modification_indic
    else:
        raise ValueError

    if model_name == 'bert':
        # Remove label values from <CLS> (first) and <SEP> (last) token
        if 'site_indic' in item:
            item['site_indic'] = item['site_indic'][1:-1]
        if 'modification_indic' in item:
            item['modification_indic'] = item['modification_indic'][1:-1]
    elif model_name == 'xlnet':
        # Remove label values from <CLS> (last) and <SEP> (second to last) token
        if 'site_indic' in item:
            item['site_indic'] = item['site_indic'][:-2]
        if 'modification_indic' in item:
            item['modification_indic'] = item['modification_indic'][:-2]
    else:
        raise NotImplementedError

    return item


if __name__ == "__main__":
    import pickle
    import pathlib

    from transformers import BertModel, AutoTokenizer, XLNetModel, XLNetTokenizer, AlbertModel, AlbertTokenizer
    from tape import TAPETokenizer, ProteinBertModel
    from tape.datasets import ProteinnetDataset, SecondaryStructureDataset

    from protein_attention.datasets import BindingSiteDataset, ProteinModificationDataset
    from protein_attention.utils import get_cache_path, get_data_path
    from protein_attention.attention_analysis.features import AminoAcidFeature, SecStructFeature, BindingSiteFeature, \
        ContactMapFeature, ProteinModificationFeature

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', required=True, help='Name of experiment. Used to create unique filename.')
    parser.add_argument('--features', nargs='+', required=True, help='list of features')
    parser.add_argument('--dataset', required=True, help='Dataset id')
    parser.add_argument('--num-sequences', type=int, required=True, help='Number of sequences to analyze')
    parser.add_argument('--model', default='bert', help='Name of model.')
    parser.add_argument('--model-version', help='Name of model version.')
    parser.add_argument('--model_dir', help='Optional directory where pretrained model is located')
    parser.add_argument('--shuffle', action='store_true', help='Whether to randomly shuffle data')
    parser.add_argument('--max-seq-len', type=int, required=True, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=123, help='PyTorch seed')
    parser.add_argument('--min-attn', type=float, help='min attention value for inclusion in analysis')
    parser.add_argument('--no_cuda', action='store_true', help='CPU only')
    args = parser.parse_args()
    print(args)

    if args.model_version and args.model_dir:
        raise ValueError('Cannot specify both model version and directory')

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
    elif args.dataset == 'protein_modifications':
        dataset = ProteinModificationDataset(get_data_path(), 'train')
    else:
        raise ValueError(f"Invalid dataset id: {args.dataset}")

    if not args.num_sequences:
        raise NotImplementedError

    if args.model == 'bert':
        if args.model_dir:
            model_version = args.model_dir
        else:
            model_version = args.model_version or 'bert-base'
        if model_version == 'prot_bert_bfd':
            model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", output_attentions=True)
            tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        elif model_version == 'prot_bert':
            model = BertModel.from_pretrained("Rostlab/prot_bert", output_attentions=True)
            tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        elif model_version == 'prot_albert':
            model = AlbertModel.from_pretrained("Rostlab/prot_albert", output_attentions=True)
            tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
        else:
            model = ProteinBertModel.from_pretrained(model_version, output_attentions=True)
            tokenizer = TAPETokenizer()
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
    elif args.model == 'xlnet':
        model_version = args.model_version
        if model_version == 'prot_xlnet':
            model = XLNetModel.from_pretrained("Rostlab/prot_xlnet", output_attentions=True)
            tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
        else:
            raise ValueError('Invalid model version')
        num_layers = model.config.n_layer
        num_heads = model.config.n_head
    else:
        raise ValueError(f"Invalid model: {args.model}")

    print('Layers:', num_layers)
    print('Heads:', num_heads)
    if cuda:
        model.to('cuda')

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
        elif feature_name == 'protein_modifications':
            features.append(ProteinModificationFeature())
        elif feature_name == 'contact_map':
            features.append(ContactMapFeature())
        else:
            raise ValueError(f"Invalid feature name: {feature_name}")

    feature_to_weighted_sum, weight_total = compute_mean_attention(
        model,
        num_layers,
        num_heads,
        items,
        features,
        tokenizer,
        args.model,
        model_version,
        cuda,
        max_seq_len=args.max_seq_len,
        min_attn=args.min_attn)

    cache_dir = get_cache_path()
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{args.exp_name}.pickle'
    pickle.dump((args, dict(feature_to_weighted_sum), weight_total), open(path, 'wb'))
    print('Wrote to', path)
