import re
from collections import defaultdict
from glob import glob
import json
from math import sqrt

import matplotlib.pyplot as plt


import numpy as np
import argparse

from matplotlib import RcParams

from src.scripts.get_epoch_graph import get_epoch_wide_results


performance_per_dataset_setting = {
    'income': {
        'SAINT': [(0.74, 0.03), (0.65, 0.15), (0.79, 0.03), (0.81, 0.03), (0.84, 0.02), (0.84, 0.02), (0.87, 0.01), (0.88, 0.00), (0.91, 0.00)],
        'TabNet': [(0.56, 0.04), (0.59, 0.07), (0.62, 0.11), (0.64, 0.06), (0.71, 0.04), (0.73, 0.05), (0.80, 0.02), (0.83, 0.02), (0.92, 0.00)],
        'NODE': [(0.54, 0.02), (0.54, 0.04), (0.65, 0.04), (0.67, 0.03), (0.75, 0.02), (0.78, 0.01), (0.78, 0.01), (0.83, 0.01), (0.82, 0.00)],
        'Logistic regression': [(0.68, 0.15), (0.72, 0.13), (0.80, 0.03), (0.82, 0.01), (0.83, 0.03), (0.85, 0.01), (0.87, 0.01), (0.88, 0.00), (0.90, 0.00)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.78, 0.03), (0.81, 0.03), (0.87, 0.01), (0.88, 0.00), (0.93, 0.00)],
        'XGBoost': [(0.50, 0.00), (0.59, 0.06), (0.77, 0.02), (0.79, 0.03), (0.82, 0.02), (0.84, 0.01), (0.87, 0.01), (0.88, 0.00), (0.93, 0.00)],
    },
    'car': {
        'SAINT': [(0.56, 0.08), (0.64, 0.08), (0.76, 0.03), (0.85, 0.03), (0.92, 0.02), (0.96, 0.01), (0.98, 0.01), (0.99, 0.00), (1.00, 0.00)],
        'TabNet': [(0.50, 0.00), (0.54, 0.05), (0.64, 0.05), (0.66, 0.05), (0.73, 0.07), (0.81, 0.04), (0.93, 0.02), (0.98, 0.01), (1.00, 0.00)],
        'NODE': [(0.51, 0.10), (0.57, 0.06), (0.69, 0.02), (0.74, 0.03), (0.80, 0.02), (0.82, 0.01), (0.91, 0.01), (0.96, 0.01), (0.93, 0.01)],
        'Logistic regression': [(0.61, 0.02), (0.65, 0.10), (0.74, 0.07), (0.83, 0.02), (0.93, 0.02), (0.96, 0.01), (0.97, 0.01), (0.98, 0.00), (0.98, 0.00)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.85, 0.06), (0.93, 0.01), (0.98, 0.01), (0.99, 0.01), (1.00, 0.00)],
        'XGBoost': [(0.50, 0.00), (0.59, 0.04), (0.70, 0.08), (0.82, 0.03), (0.91, 0.02), (0.95, 0.01), (0.98, 0.01), (0.99, 0.01), (1.00, 0.00)],
    },
    'heart': {
        'SAINT': [(0.80, 0.12), (0.83, 0.10), (0.88, 0.07), (0.90, 0.01), (0.90, 0.04), (0.90, 0.02), (0.90, 0.01), (0.92, 0.01), (0.93, 0.01)],
        'TabNet': [(0.56, 0.12), (0.70, 0.05), (0.73, 0.14), (0.80, 0.04), (0.83, 0.05), (0.84, 0.03), (0.88, 0.02), (0.88, 0.03), (0.89, 0.03)],
        'NODE': [(0.52, 0.10), (0.78, 0.08), (0.83, 0.03), (0.86, 0.02), (0.88, 0.02), (0.88, 0.01), (0.91, 0.02), (0.92, 0.03), (0.92, 0.03)],
        'Logistic regression': [(0.69, 0.17), (0.75, 0.13), (0.82, 0.06), (0.87, 0.05), (0.91, 0.01), (0.90, 0.02), (0.92, 0.01), (0.93, 0.01), (0.93, 0.01)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.91, 0.01), (0.91, 0.01), (0.91, 0.01), (0.93, 0.00), (0.94, 0.01)],
        'XGBoost': [(0.50, 0.00), (0.55, 0.14), (0.84, 0.07), (0.88, 0.04), (0.91, 0.01), (0.91, 0.01), (0.90, 0.01), (0.92, 0.01), (0.94, 0.01)],
    },
    'diabetes': {
        'SAINT': [(0.46, 0.12), (0.65, 0.11), (0.73, 0.06), (0.73, 0.06), (0.79, 0.03), (0.81, 0.03), (0.81, 0.04), (0.77, 0.03), (0.83, 0.03)],
        'TabNet': [(0.56, 0.04), (0.56, 0.06), (0.64, 0.09), (0.66, 0.06), (0.71, 0.04), (0.73, 0.04), (0.74, 0.05), (0.74, 0.07), (0.81, 0.03)],
        'NODE': [(0.49, 0.13), (0.67, 0.09), (0.69, 0.08), (0.73, 0.05), (0.77, 0.04), (0.80, 0.04), (0.81, 0.03), (0.83, 0.02), (0.83, 0.03)],
        'Logistic regression': [(0.60, 0.15), (0.68, 0.11), (0.73, 0.05), (0.76, 0.05), (0.80, 0.02), (0.81, 0.02), (0.83, 0.02), (0.83, 0.02), (0.83, 0.02)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.79, 0.02), (0.79, 0.04), (0.79, 0.02), (0.79, 0.03), (0.83, 0.03)],
        'XGBoost': [(0.50, 0.00), (0.59, 0.16), (0.72, 0.07), (0.69, 0.08), (0.73, 0.05), (0.78, 0.05), (0.80, 0.03), (0.80, 0.01), (0.84, 0.03)],
    }
}

test_set_sizes_per_dataset = {
    'income': 1,  # 9769,
    'car': 1,  # 346,
    'heart': 1,  # 184,
    'diabetes': 1,  # 154
}


def create_performance_graph(args):
    shots = [0, 4, 8, 16, 32, 64, 128, 256, 512]
    total_samples = sum(test_set_sizes_per_dataset.values())
    markers = ['x', 's', 'd', 'o', '^', 'p', 'P', '8']

    # Determine means and sd for each dataset
    fig, ax = plt.subplots(figsize=(6, 5))
    # TODO
    # plt.rcParams["font.family"] = "Times"
    included_performance_scores = ['Logistic regression', 'LightGBM', 'XGBoost', 'SAINT', 'TabNet', 'NODE']

    # Create list of performance results for each setting
    for k, setting in enumerate(included_performance_scores):
        weighted_mean = [0.] * len(shots)
        weighted_variance = [0.] * len(shots)

        for dataset in test_set_sizes_per_dataset.keys():
            for i, shot in enumerate(shots):
                weighted_mean[i] += test_set_sizes_per_dataset[dataset] * performance_per_dataset_setting[dataset][setting][i][0]
                weighted_variance[i] += test_set_sizes_per_dataset[dataset] * (performance_per_dataset_setting[dataset][setting][i][1] ** 2)

        means = np.array(weighted_mean) / total_samples
        stds = np.sqrt(np.array(weighted_variance)) / total_samples

        ax.plot(range(0, len(shots)), means, marker=markers[k], label=setting)
        ax.fill_between(range(0, len(shots)), (means - stds), (means + stds), alpha=.1)
        ax.set_ylabel('Average AUC across tabular datasets')
        ax.set_xlabel('Number of shots')
        ax.legend(loc='lower right')
        # ax.set_xscale('symlog', base=2)
        ax.set_xlim(xmin=0, xmax=len(shots) - 1)
        ax.set_xticks(range(0, len(shots)), [str(s) for s in shots])
        ax.set_ylim(ymin=0.45, ymax=1.)

    plt.tight_layout()
    plt.show()
    # plt.savefig('performance_graph.png', dpi=600)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    create_performance_graph(args)


