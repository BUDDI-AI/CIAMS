# standard libraries
import logging
import multiprocessing
import os
# import warnings

# third party libraries
import numpy as np
from tqdm import tqdm

# local application modules
from .utils import read_automs_config, generate_warehouse_path_default, generate_oneshot_default, generate_num_processes_default, _check_automs_parameters, _check_dataset_filename, read_dataset_config_file, _check_dataset, create_current_run_warehouse_dir, write_current_run_metainfo, write_classification_complexity_to_file, write_f1_scores_to_file, write_consolidated_results_to_file
from .sampling import read_dataset_file, sample_dataset
from .cluster_indices_generation import bag_generate_cluster_indices, convert_cluster_indices_to_features
from .classification_complexity_prediction import predict_classification_complexity
from .f1_scores_estimation import estimate_f1_scores
from .true_f1_scores_computation import compute_true_f1_scores

# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# setup logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S',
            level = logging.INFO)
logger = logging.getLogger(__name__)

AUTOMS_WAREHOUSE_PATH, AUTOMS_APPROACH, AUTOMS_NUM_PROCESSES = read_automs_config()
warehouse_path_default = generate_warehouse_path_default(AUTOMS_WAREHOUSE_PATH)
oneshot_default = generate_oneshot_default(AUTOMS_APPROACH)
num_processes_default = generate_num_processes_default(AUTOMS_NUM_PROCESSES)

def automs(dataset_filename, oneshot=oneshot_default, num_processes=num_processes_default, warehouse_path=warehouse_path_default, return_true_f1s=False):
    """ Predicts the maximum achievable F1-score corresponding to various classifier models for a given dataset

    Parameters
    ----------
    dataset_filename : str
        path to a CSV, LIBSVM or ARFF data file. The path must also have associated configuration file for the dataset (with name as `dataset_filename` suffixed with '.config.py').
    oneshot : bool, optional
        Whether to use oneshot approach or subsampling approach (default corresponds to the 'approach' specified in automs configuration, if automs is configured. Else, it defaults to True).
    num_processes : int, optional
        Number of parallel processes or jobs to use (default corresponds to 'num processes' specified in automs configuration, if automs is configured. Else, it defaults to 1).
    warehouse_path : str, optional
        Location for storing intermediate files and results corresponding to the dataset being processed (default corresponds to 'warehouse path' specified in automs configuration, if automs is configured. Else, it must be specified here).
    return_true_f1s : bool, optional
        Whether to return the true f1-scores for dataset. If True, also returns the true f1-scores for dataset.

    Returns
    -------
    bool
        Classification complexity of dataset. True implies dataset is hard to classify, False implies dataset is not hard to classify.
    dict of {str : int}
        Dictionary mapping each classification model name to its estimated f1 score.
    dict of {str : int}, optional
        Dictionary mapping each classification model name to its true f1 score. Only returned if `return_true_f1s` is True.

    Raises
    ------
    ValueError
        If the dataset is not a binary classification dataset.
    ValueError
        If oneshot is specified to False, and the dataset has less or equal to 500 samples.
    
    Notes
    -----
    AutoMS performs cluster analysis and generates cluster indices on the dataset or its subsamples, and estimates the f1-scores for various classification models from these clustering-based metafeatures by using fitted regression models.

    AutoMS works only for binary classification datasets, although the idea can be extended to multi-class problems using one-vs-rest statergy.

    AutoMS with sub-sampling approach works only for datasets with more than 500 samples.
    """
    logger.info("Parameters : dataset_filename = '%s', oneshot = %s, num_processes = %s, warehouse_path = '%s'", dataset_filename, oneshot, num_processes, warehouse_path)

    # check the type and value of automs parameters
    _check_automs_parameters(dataset_filename, oneshot, num_processes, warehouse_path)

    # identify the dataset filetype and validate if it is supported
    logger.info("Running dataset : '%s'", dataset_filename)
    dataset_filetype = _check_dataset_filename(dataset_filename)
    logger.info("Dataset filetype : %s", dataset_filetype)

    # read the dataset configuration file
    dataset_config_filename = dataset_filename + '.config.py'
    dataset_config = read_dataset_config_file(dataset_config_filename, dataset_filetype)
    logger.info("Dataset configuration : %s", vars(dataset_config))

    # read the dataset file with the read dataset configuration
    dataset = read_dataset_file(dataset_filename, dataset_config)
    logger.info("Number of samples = %d, Number of features (after dummy coding) = %d", dataset.n_samples, dataset.n_features)

    # check if the dataset meets the requirements of automs
    _check_dataset(dataset)

    # create the warehouse directory for the current run and write meta information to it
    dataset_base_filename = os.path.basename(dataset_filename)
    is_user_configured_warehouse = (warehouse_path == AUTOMS_WAREHOUSE_PATH)
    current_run_warehouse_dir = create_current_run_warehouse_dir(warehouse_path, dataset_base_filename, is_user_configured_warehouse)
    write_current_run_metainfo(current_run_warehouse_dir, dataset_filename, dataset_config, oneshot=oneshot, num_processes=num_processes)
    logger.info("Current run warehouse directory : %s", current_run_warehouse_dir)

    # sample the dataset into stratified subsamples
    sampling_results = sample_dataset(dataset, oneshot, current_run_warehouse_dir)
    bags_filenames = sampling_results['bags_filenames']

    # perform clustering on the bags and generate cluster indices
    logging.getLogger('automs.eda').setLevel(logging.WARN) # Reduce logging level for `automs.eda`

    if oneshot:
        bag_filename = bags_filenames[0]
        bag_cluster_indices = bag_generate_cluster_indices(bag_filename, n_jobs=num_processes)
        bags_cluster_indices = [bag_cluster_indices]

    else:
        process_pool = multiprocessing.Pool( processes = min(num_processes, len(bags_filenames)) )
        bags_cluster_indices = list(tqdm(process_pool.imap(bag_generate_cluster_indices, bags_filenames), total=len(bags_filenames), desc="Number of bags processed"))
        # bags_cluster_indices = process_pool.map(bag_generate_cluster_indices, bags_filenames)
        process_pool.close()
        process_pool.join()

    # Convert the bags' cluster indices into feature vectors
    bags_feature_vectors = list(map(convert_cluster_indices_to_features, bags_cluster_indices))
    logger.info("Number of clustering-based metafeatures extracted from each bag = %d", len(bags_feature_vectors[0]))

    # Predict the bags' classification complexity (hardness of classification)
    bags_predicted_clf_complexity = predict_classification_complexity(bags_feature_vectors)
    # average the clf complexity boolean value across bags to predict clf complexity boolean value for dataset
    predicted_clf_complexity = ( np.mean(bags_predicted_clf_complexity) >= 0.5 )
    logger.info("Predicted classification complexity for dataset = %s", 'IS HARD TO CLASSIFY' if predicted_clf_complexity else 'IS NOT HARD TO CLASSIFY')

    # Write predicted classification complexity tp file
    predicted_clf_complexity_filename = os.path.join(current_run_warehouse_dir, 'predicted_classification_complexity')
    write_classification_complexity_to_file(predicted_clf_complexity, predicted_clf_complexity_filename)

    # Estimate the bags' f1 score for different classifier models
    clf_models_bags_estimated_f1_scores = estimate_f1_scores(bags_feature_vectors)
    # average the classifier model's f1 score across bags to estimate f1 score for dataset
    clf_models_estimated_f1_scores = { clf_model : np.mean(bags_estimated_f1_score) for clf_model, bags_estimated_f1_score in clf_models_bags_estimated_f1_scores.items() }
    logger.info("Estimated f1-scores for dataset = %s", clf_models_estimated_f1_scores)

    # write dataset's estimated f1-scores to file
    estimated_f1_scores_filename = os.path.join(current_run_warehouse_dir, 'estimated_f1_scores')
    write_f1_scores_to_file(clf_models_estimated_f1_scores, estimated_f1_scores_filename)

    # write consolidated results to file
    consolidated_results_filename = os.path.join(current_run_warehouse_dir, 'results.xlsx')
    write_consolidated_results_to_file(bags_cluster_indices, bags_predicted_clf_complexity, predicted_clf_complexity, clf_models_bags_estimated_f1_scores, clf_models_estimated_f1_scores, consolidated_results_filename)

    if return_true_f1s:
        clf_models_true_f1_scores = compute_true_f1_scores(dataset)
        logger.info("True f1-scores for dataset = %s", clf_models_true_f1_scores)

        # write dataset's estimated f1-scores to file
        true_f1_scores_filename = os.path.join(current_run_warehouse_dir, 'true_f1_scores')
        write_f1_scores_to_file(clf_models_true_f1_scores, true_f1_scores_filename)

        return predicted_clf_complexity, clf_models_estimated_f1_scores, clf_models_true_f1_scores

    return predicted_clf_complexity, clf_models_estimated_f1_scores

