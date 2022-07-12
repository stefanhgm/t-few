import re
from collections import defaultdict
from glob import glob
import json
import matplotlib.pyplot as plt

import numpy as np
import os
import argparse


def make_epoch_graph(args):
    def collect_exp_scores(exp_name_template, datasets):
        print("=" * 80)
        all_files = glob(
            os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), exp_name_template, "dev_scores.json")
        )
        print(f"Find {len(all_files)} experiments fit into {exp_name_template}")

        def read_acc_per_epoch(fname):
            results = []
            with open(fname) as f:
                for l in f.readlines():
                    e = json.loads(l)
                    results.append(e["AUC"])
            return results

        results = defaultdict(list)
        all_files.sort(key=lambda x: (int(re.findall(r".*_numshot(\d+).*", x)[0]), x), reverse=True)
        for fname in all_files:
            if any([d in fname for d in datasets]):
                name = fname.split('_seed')[0]
                accs = read_acc_per_epoch(fname)
                if (len(accs) not in [41, 81, 25, 49]) or (len(results[name]) > 0 and len(results[name][0]) > len(accs)):
                    continue  # Ignore incomplete results
                results[name] = results[name] + [read_acc_per_epoch(fname)]

        # Determine means and sd for each dataset
        fig, ax = plt.subplots(figsize=(8, 6))
        epoch_steps = 5
        epoch_result = 29
        print(f"Use the {epoch_result}th epoch for the results (indexed by 0 so should be +1).")
        for k, v in results.items():
            means = np.mean(np.array(v), axis=0)
            stds = np.std(np.array(v), axis=0)
            epochs = [0] + list(range(epoch_steps - 1, (len(means) * epoch_steps) - 1, epoch_steps))
            ax.plot(epochs, means, label=k)
            ax.fill_between(epochs, (means - stds), (means + stds), alpha=.1)
            print(f"{k}: {means[epochs.index(epoch_result)] * 100:.2f} ({stds[epochs.index(epoch_result)] * 100:.2f})")
        plt.legend(loc='lower right')
        plt.xlabel('steps of 5 epochs')
        plt.ylabel('AUC')
        plt.ylim(ymin=0.35, ymax=1.)
        plt.tight_layout()
        plt.show()
        print()

    for exp_name_template in args.exp_name_templates:
        collect_exp_scores(exp_name_template, args.datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name_templates", default="t03b_*_finetune", required=True)
    parser.add_argument("-d", "--datasets", required=True)
    args = parser.parse_args()
    args.exp_name_templates = args.exp_name_templates.split(",")
    args.datasets = args.datasets.split(",")
    make_epoch_graph(args)
