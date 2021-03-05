import argparse

from .automs import automs, warehouse_path_default, oneshot_default, num_processes_default

def get_parser():

    parser = argparse.ArgumentParser(description="Automatic Model Selection Using Cluster Indices")

    parser.add_argument('dataset_filename', type=str, help="path to a CSV, LIBSVM or ARFF data file. The path must also have associated configuration file for the dataset (with name as `dataset_filename` suffixed with '.config.py').")

    approach_group = parser.add_mutually_exclusive_group()
    approach_group.add_argument('--oneshot', action='store_true', help="Whether to use oneshot approach")
    approach_group.add_argument('--subsampling', action='store_true', help="Whether to use sub-sampling approach")

    parser.add_argument('--num_processes', type=int, default=num_processes_default, help="Number of parallel processes or jobs to use")
    parser.add_argument('--warehouse_path', type=str, default=warehouse_path_default, help="Location for storing intermediate files and results corresponding to the dataset being processed")

    parser.add_argument('--truef1', action='store_true', help="Whether to compute the true f1-scores for dataset")

    parser.add_argument('--result', type=str, metavar='RESULTS_FILENAME', help="Path to file to write predicted classification complexity, estimated f1 scores and true f1 scores results")

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    # postprocess the 'oneshot' and 'subsampling' arguments
    oneshot = oneshot_default

    if args.oneshot: oneshot = True
    elif args.subsampling: oneshot = False

    outputs = automs(args.dataset_filename, oneshot=oneshot, num_processes=args.num_processes, warehouse_path=args.warehouse_path, return_true_f1s=args.truef1)

    print(f"Predicted Classification complexity for dataset = {'IS HARD TO CLASSIFY' if outputs[0] else 'IS NOT HARD TO CLASSIFY'}")
    print(f"Estimated f1-scores for dataset = {outputs[1]}")

    if args.truef1:
        print(f"True f1-scores for dataset = {outputs[2]}")

    # write the estimated f1 scores to results file
    if args.result:

        with open(args.result, 'w') as f_results:

            # write predicted classification complexity to results file
            is_hard_to_classify = outputs[0]
            f_results.write(f"[CLASSIFICATION COMPLEXITY]\nis hard to classify = {is_hard_to_classify}\n\n")

            # write estimated f1 scores to results file
            clf_models_estimated_f1_scores = outputs[1]
            f_results.write("[ESTIMATED F1 SCORES]\n")
            for clf_model, estimated_f1_score in clf_models_estimated_f1_scores.items():
                f_results.write(f"{clf_model} = {estimated_f1_score}\n")
            f_results.write("\n")

            # write computed true f1 scores to results file
            if args.truef1:
                clf_models_true_f1_scores = outputs[2]
                f_results.write("[TRUE F1 SCORES]\n")
                for clf_model, true_f1_score in clf_models_true_f1_scores.items():
                    f_results.write(f"{clf_model} = {true_f1_score}\n")
                f_results.write("\n")

