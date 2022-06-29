from glob import glob
import json

import pandas as pd
import os
import argparse


def make_result_table(args):
    def collect_exp_scores(exp_name_template, datasets):
        print("=" * 80)
        all_files = glob(
            os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), exp_name_template, "dev_scores.json")
        )
        print(f"Find {len(all_files)} experiments fit into {exp_name_template}")

        def read_last_eval(fname):
            with open(fname) as f:
                e = json.loads(f.readlines()[-1])
            return e["accuracy"]

        def read_results(fname):
            with open(fname) as f:
                # Determine index of maximum row with max AUC and second criteria max accuracy
                e = [json.loads(l) for l in f.readlines()]
                max_idx = max(range(len(e)), key=lambda i: (e[i].get('AUC', 0), e[i].get('accuracy', 0)))
                results = e[max_idx]
                results = {**results, 'best_eval_step': max_idx + 1, 'eval_steps': len(e)}
                results = {**results, 'last_AUC': e[-1].get('AUC', 0), 'last_accuracy': e[-1].get('accuracy', 0)}
                return results

        def parse_expname(fname):
            elements = fname.split("/")[-2].split("_")
            results = {}
            results['model'] = elements[0]
            results['dataset'] = elements[1]
            numshot = int(([e for e in elements if e.startswith('numshot')][0])[len('numshot'):])
            elements.remove('numshot' + str(numshot))
            seed = int(([e for e in elements if e.startswith('seed')][0])[len('seed'):])
            elements.remove('seed' + str(seed))
            results['spec'] = '_'.join([e for e in elements if e not in results.values()])
            results['numshot'] = numshot
            results['seed'] = seed
            # return tuple(elements[:3] + ["_".join(elements[3:])])
            return results

        def read_test_results(fname):
            fname = fname.replace('dev_scores.json', 'test_scores.json')
            if not os.path.exists(fname):
                return {}
            with open(fname) as f:
                results = json.loads(f.readlines()[-1])
                keys = list(results.keys())
                for k in keys:
                    results[k + '_test'] = results.pop(k)
            return results

        results = []
        for fname in all_files:
            result = {**(parse_expname(fname)), **(read_results(fname)), **(read_test_results(fname))}
            results.append(result)
            # Return all results collected for one type of datasets
        return results

    results = []
    for exp_name_template in args.exp_name_templates:
        results += collect_exp_scores(exp_name_template, args.datasets)

    output_fname = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), "overview.csv")
    output_csv = pd.DataFrame(results)
    output_csv = output_csv.sort_values(['model', 'dataset', 'spec', 'numshot'])
    output_csv.to_csv(output_fname, float_format='%.4f')
    print(f"Save result to {output_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name_templates", default="t03b_*_finetune", required=True)
    parser.add_argument(
        "-d", "--datasets", default="copa,h-swag,storycloze,winogrande,wsc,wic,rte,cb,anli-r1,anli-r2,anli-r3"
    )
    args = parser.parse_args()
    args.exp_name_templates = args.exp_name_templates.split(",")
    args.datasets = args.datasets.split(",")
    make_result_table(args)
