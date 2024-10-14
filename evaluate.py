
import yaml
import argparse

from evaluation import run_ms_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cl', type=int, default=-1)
    parser.add_argument('--arch', type=str, default='null')
    parser.add_argument('--p', type=float, default=-1)
    parser.add_argument('--k', type=int, default=-1)
    parser.add_argument('--e', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='null')
    parser.add_argument('--folder', type=str, default='null')
    args = parser.parse_args()

    with open("params.yml", 'r') as f:
        params = yaml.safe_load(f)

    params["pca_variance_percentage"] = args.p if args.p != -1 else params["pca_variance_percentage"]
    params["class_label"] = args.cl if args.cl != -1 else params["class_label"]
    params["dataset_file"] =  args.dataset if args.dataset != 'null' else params["dataset_file"]
    params["architecture"] =  args.arch if args.arch != 'null' else params["architecture"]
    params["max_epochs"] = args.e if args.e != -1 else params["max_epochs"]
    params["k"] = args.k if args.k != -1 else params["k"]
    params["load_from"] = args.folder if args.folder != 'null' else params["load_from"]

    # Run in a single node
    run_ms_eval(0, params)


if __name__ == "__main__":
    main()
