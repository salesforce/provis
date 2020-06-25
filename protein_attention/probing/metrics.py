"""Metrics for evaluating probing classifiers

Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from typing import Sequence, Union

import numpy as np
from scipy.special import softmax


def precision(target: Union[Sequence[int], Sequence[Sequence[int]]],
              prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        raise NotImplementedError
    else:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            is_incorrect = ~is_correct

            is_predicted_true = pred_array[mask] == 1
            is_predicted_false = ~is_predicted_true

            tp += (is_predicted_true & is_correct).sum()
            fp += (is_predicted_true & is_incorrect).sum()
            tn += (is_predicted_false & is_correct).sum()
            fn += (is_predicted_false & is_incorrect).sum()

        print('tp:', tp, 'fp:', fp, 'tn:', tn, 'fn:', fn)
        return tp / (tp + fp)


def recall(target: Union[Sequence[int], Sequence[Sequence[int]]],
           prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        raise NotImplementedError
    else:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            is_incorrect = ~is_correct

            is_predicted_true = pred_array[mask] == 1
            is_predicted_false = ~is_predicted_true

            tp += (is_predicted_true & is_correct).sum()
            fp += (is_predicted_true & is_incorrect).sum()
            tn += (is_predicted_false & is_correct).sum()
            fn += (is_predicted_false & is_incorrect).sum()

        print('tp:', tp, 'fp:', fp, 'tn:', tn, 'fn:', fn)
        return tp / (tp + fn)


def f1(target: Union[Sequence[int], Sequence[Sequence[int]]],
       prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    p = precision(target, prediction)
    r = recall(target, prediction)
    return 2 * p * r / (p + r)


def precision_at_ks(ks: Sequence[int], target: Union[Sequence[int], Sequence[Sequence[int]]],
                    prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        raise NotImplementedError
    else:
        top_k_all = []
        for k, label, score in zip(ks, target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score)
            num_classes = pred_array.shape[-1]
            if num_classes != 2:
                raise NotImplementedError('Currently only support binary classification tasks')
            probs = softmax(pred_array, axis=-1)
            pos_probs = probs[:, 1]
            mask = label_array != -1
            score_labels = []
            num_pos = 0
            num_total = 0

            for label, pos_prob, m in zip(label_array, pos_probs, mask):
                if m:
                    score_labels.append((pos_prob, label))
                    num_total += 1
                    if label == 1:
                        num_pos += 1
                    if label not in (0, 1):
                        print(label)
                    # print('added', (score, label))
            if len(score_labels) == 0:
                continue
            top = sorted(score_labels, reverse=True)[:k]
            top_labels = list(zip(*top))[1]
            top_k_all.extend(top_labels)
        return sum(top_k_all) / len(top_k_all)


if __name__ == '__main__':

    target = [
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        np.array([1, 0, 1, 0, 1, 0, 1, 1])]
    prediction = [
        np.array([[.2, .8], [0., .2], [0., 1], [.1, .9], [.9, .1], [1., 0.], [0., 1.], [0., .3], [0., .7], [0., .2],
                  [0., 1.], [0., 0.], [0., 0.], [0., 0.]]),
        np.array([[0., .9], [0., .1], [0., .1], [0., .8], [0., .9], [0., 0], [0., 0]])
    ]
    ks = [
        6,  # 5/6 are correct
        3,  # 2/3 are correct
    ]

    assert precision_at_ks(ks, target, prediction) == (5 + 2) / (6 + 3)
