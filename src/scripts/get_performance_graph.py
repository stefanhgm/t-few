import re
from collections import defaultdict
from glob import glob
import json
import matplotlib.pyplot as plt

import numpy as np
import argparse

from src.scripts.get_epoch_graph import get_epoch_wide_results


def make_epoch_graph(args):
    def make_epoch_graph_per_template(exp_name_template, datasets):
        results = get_epoch_wide_results(exp_name_template, datasets)
        # Determine means and sd for each dataset
        fig, ax = plt.subplots(figsize=(8, 6))
        epoch_steps = 10
        epoch_result = 0
        print(f"Use the {epoch_result}th epoch for the results (indexed by 0 so should be +1).")
        setting_dict = {
            'list_': 'list',
            'list_values_': 'list only values',
            # 'list_shuffled_': 'list permuted values',
            '': 'natural text',
            'Logistic regression': 'Logistic regression',
            'LightGBM': 'LightGBM',
            'XGBoost': 'XGBoost'
        }

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax_list = ax.flatten()
        # Create list of performance results for each setting
        # Manually input zero-shot from get_epoch_graph setting epoch_results to 0
        # Manually inout baseline results
        performance_per_dataset_setting = {
            'income': {
                'list': [(0, 0.80, 0.01)],
                'list only values': [(0, 0.73, 0.01)],
                # 'list permuted values': [(0, 0.50, 0.01)],
                'natural text': [(0, 0.74, 0.01)],
                'Logistic regression': [(0, 0.50, 0.00), (4, 0.68, 0.15), (8, 0.72, 0.13), (16, 0.80, 0.03), (32, 0.82, 0.01),
                                        (64, 0.83, 0.03), (128, 0.85, 0.01), (256, 0.87, 0.01), (512, 0.88, 0.00)],
                'LightGBM': [(0, 0.50, 0.00), (4, 0.64, 0.13), (8, 0.66, 0.13), (16, 0.78, 0.04), (32, 0.81, 0.02),
                             (64, 0.81, 0.03), (128, 0.84, 0.02), (256, 0.87, 0.01), (512, 0.88, 0.01)],
                'XGBoost': [(0, 0.50, 0.00), (4, 0.59, 0.13), (8, 0.64, 0.12), (16, 0.77, 0.05), (32, 0.79, 0.03),
                            (64, 0.80, 0.03), (128, 0.83, 0.02), (256, 0.87, 0.01), (512, 0.88, 0.01)
]
            },
            'car': {
                'list': [(0, 0.79, 0.02)],
                'list only values': [(0, 0.48, 0.03)],
                # 'list permuted values': [(0, 0.53, 0.02)],
                'natural text': [(0, 0.80, 0.02)],
                'Logistic regression': [(0, 0.50, 0.00), (4, 0.61, 0.02), (8, 0.65, 0.10), (16, 0.74, 0.07), (32, 0.83, 0.02),
                                        (64, 0.93, 0.02), (128, 0.96, 0.01), (256, 0.97, 0.01), (512, 0.98, 0.00)],
                'LightGBM': [(0, 0.50, 0.00), (4, 0.60, 0.05), (8, 0.66, 0.09), (16, 0.74, 0.07), (32, 0.84, 0.03),
                             (64, 0.93, 0.02), (128, 0.96, 0.01), (256, 0.98, 0.01), (512, 0.98, 0.01)],
                'XGBoost': [(0, 0.50, 0.00), (4, 0.56, 0.06), (8, 0.64, 0.09), (16, 0.72, 0.09), (32, 0.83, 0.03),
                            (64, 0.93, 0.02), (128, 0.95, 0.01), (256, 0.98, 0.01), (512, 0.99, 0.01)]
            },
            'heart': {
                'list': [(0, 0.49, 0.03)],
                'list only values': [(0, 0.40, 0.04)],
                # 'list permuted values': [(0, 0.50, 0.04)],
                'natural text': [(0, 0.52, 0.04)],
                'Logistic regression': [(0, 0.50, 0.00), (4, 0.69, 0.17), (8, 0.75, 0.13), (16, 0.82, 0.06), (32, 0.87, 0.05),
                                        (64, 0.91, 0.01), (128, 0.90, 0.02), (256, 0.92, 0.01), (512, 0.93, 0.01)],
                'LightGBM': [(0, 0.50, 0.00), (4, 0.62, 0.18), (8, 0.70, 0.16), (16, 0.82, 0.06), (32, 0.87, 0.05),
                             (64, 0.90, 0.02), (128, 0.90, 0.01), (256, 0.91, 0.02), (512, 0.93, 0.01)],
                'XGBoost': [(0, 0.50, 0.00), (4, 0.58, 0.15), (8, 0.65, 0.17), (16, 0.82, 0.06), (32, 0.87, 0.04),
                            (64, 0.90, 0.02), (128, 0.90, 0.01), (256, 0.91, 0.01), (512, 0.92, 0.01)]
            },
            'diabetes': {
                'list': [(0, 0.63, 0.07)],
                'list only values': [(0, 0.55, 0.05)],
                # 'list permuted values': [(0, 0.48, 0.03)],
                'natural text': [(0, 0.76, 0.03)],
                'Logistic regression': [(0, 0.50, 0.00), (4, 0.60, 0.15), (8, 0.68, 0.11), (16, 0.73, 0.05), (32, 0.76, 0.05),
                                        (64, 0.80, 0.02), (128, 0.81, 0.02), (256, 0.83, 0.02), (512, 0.83, 0.02)],
                'LightGBM': [(0, 0.50, 0.00), (4, 0.58, 0.14), (8, 0.66, 0.11), (16, 0.68, 0.09), (32, 0.74, 0.05),
                             (64, 0.77, 0.04), (128, 0.79, 0.05), (256, 0.82, 0.03), (512, 0.82, 0.02)],
                'XGBoost': [(0, 0.50, 0.00), (4, 0.55, 0.12), (8, 0.63, 0.13), (16, 0.69, 0.08), (32, 0.73, 0.06),
                            (64, 0.77, 0.04), (128, 0.79, 0.05), (256, 0.81, 0.03), (512, 0.81, 0.03)]
            }
        }

        for i, d in enumerate(datasets):
            performance_per_setting = performance_per_dataset_setting[d]
            for k, v in results.items():
                if d not in k:
                    continue
                means = np.mean(np.array(v), axis=0)
                stds = np.std(np.array(v), axis=0)
                epochs = [0] + list(range(epoch_steps - 1, (len(means) * epoch_steps) - 1, epoch_steps))
                # Mean and std result at fixed epoch 30
                mean = means[epochs.index(epoch_result)]
                std = stds[epochs.index(epoch_result)]
                setting, shots = re.findall(rf".+_{d}_(.*)numshot(\d+)", k)[0]
                if setting in ['list_shuffled_', 'gpt_', '2_', 'ttt_', 't0_']:
                    continue
                setting = setting_dict[setting]
                performance_per_setting[setting] = performance_per_setting[setting] + [(int(shots), mean, std)]

                # ax.plot(epochs, means, label=k)
                # ax.fill_between(epochs, (means - stds), (means + stds), alpha=.1)
                # print(f"{k}: {means[epochs.index(epoch_result)] * 100:.2f} ({stds[epochs.index(epoch_result)] * 100:.2f})")
            for setting in setting_dict.values():
                performance_per_setting[setting] = sorted(performance_per_setting[setting], key=lambda x: x[0])
                shots, means, stds = list(zip(*performance_per_setting[setting]))
                # Als round to two digits as results table.
                means = np.around(np.array(means), 2)
                stds = np.around(np.array(stds), 2)
                ax_list[i].plot(shots, means, label=setting)
                ax_list[i].fill_between(shots, (means - stds), (means + stds), alpha=.1)
                ax_list[i].set_title(d)
                if d == 'car':
                    ax_list[i].set_ylabel('Macro AUC OVR')
                else:
                    ax_list[i].set_ylabel('AUC')
                ax_list[i].legend(loc='lower right')
                if i > 1:
                    ax_list[i].set_xlabel('Number of shots')
                ax_list[i].set_xscale('symlog')
                ax_list[i].set_xlim(xmin=1, xmax=512)
                ax_list[i].set_xticks([1, 10, 100])
                ax_list[i].set_ylim(ymin=0.35, ymax=1.)

        plt.tight_layout()
        # plt.show()
        plt.savefig('performance_graph.png', dpi=600)
        print()

    for exp_name_template in args.exp_name_templates:
        make_epoch_graph_per_template(exp_name_template, args.datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name_templates", default="t03b_*_finetune", required=True)
    parser.add_argument("-d", "--datasets", required=True)
    args = parser.parse_args()
    args.exp_name_templates = args.exp_name_templates.split(",")
    args.datasets = args.datasets.split(",")
    make_epoch_graph(args)


