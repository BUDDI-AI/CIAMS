""" Helper utility functions for automs """

from collections import OrderedDict
import configparser
from datetime import datetime
import importlib.util
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from .config import DATASET_FILETYPE_CONFIG
from .eda import EDA


# setup logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S',
            level = logging.INFO)
logger = logging.getLogger(__name__)


def read_automs_config():
    """ Read the automs config file and retrieve necessary configuration values """

    AUTOMS_WAREHOUSE_PATH = None
    AUTOMS_APPROACH = None
    AUTOMS_NUM_PROCESSES = None

    config_file = os.path.expanduser('~/.config/automs.ini')
    if os.path.exists(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        config_defaults = config['DEFAULT']
        AUTOMS_WAREHOUSE_PATH = config_defaults.get('warehouse path')
        AUTOMS_APPROACH = config_defaults.get('approach')
        AUTOMS_NUM_PROCESSES = config_defaults.getint('num processes')

    else:
        logger.warning("AutoMS is not configured. Configure AutoMS by running `automs-config` on your terminal.")

    return AUTOMS_WAREHOUSE_PATH, AUTOMS_APPROACH, AUTOMS_NUM_PROCESSES


def generate_warehouse_path_default(AUTOMS_WAREHOUSE_PATH):
    """ Generates the default warehouse path parameter value to use using configured 'warehouse path' """

    if AUTOMS_WAREHOUSE_PATH is None:
        warehouse_path_default = None # default path (if any) to use if AutoMS is not configured

    else:
        warehouse_path_default = AUTOMS_WAREHOUSE_PATH

    return warehouse_path_default


def generate_oneshot_default(AUTOMS_APPROACH):
    """ Generates the default oneshot parameter value to use using configured 'approach' """

    if AUTOMS_APPROACH is None:
        oneshot_default = True

    elif AUTOMS_APPROACH == 'oneshot':
        oneshot_default = True

    elif AUTOMS_APPROACH == 'sub-sampling':
        oneshot_default = False

    else:
        logger.error(f"Invalid approach configuration value '{AUTOMS_APPROACH}' encountered. Reconfigure AutoMS by running `automs-config` on your terminal.")
        raise ValueError("Invalid approach configuration value encountered")

    return oneshot_default


def generate_num_processes_default(AUTOMS_NUM_PROCESSES):
    """ Generates the default num processes parameter value to use using configured 'num processes' """

    if AUTOMS_NUM_PROCESSES is None:
        num_processes_default = 1

    else:
        num_processes_default = AUTOMS_NUM_PROCESSES

    return num_processes_default


def _check_automs_parameters(dataset_filename, oneshot, num_processes, warehouse_path):

    if not type(dataset_filename) == str:
        logger.error(f"Encountered 'dataset_filename' parameter of type {type(dataset_filename).__name__}, while expecting type str.")
        raise TypeError(f"'dataset_filename' must be str, not {type(dataset_filename).__name__}")

    if not type(oneshot) == bool:
        logger.error(f"Encountered 'oneshot' parameter of type {type(oneshot).__name__}, while expecting type bool.")
        raise TypeError(f"'oneshot' must be bool, not {type(oneshot).__name__}")

    if not type(num_processes) == int:
        logger.error(f"Encountered 'num_processes' parameter of type {type(num_processes).__name__}, while expecting type int.")
        raise TypeError(f"'num_processes' must be bool, not {type(oneshot).__name__}")


    if oneshot and not(num_processes > 0 or num_processes == -1):
        logger.error(f"Encountered 'num_processes' parameter equal to {num_processes}, while permitted values for oneshot approach are -1 and any integer values greater than zero.")
        raise ValueError(f"'num_processes' must be -1 or int >0, not {num_processes} for oneshot approach")

    if not oneshot and not(num_processes > 0):
        logger.error(f"Encountered 'num_processes' parameter equal to {num_processes}, while permitted values for sub-sampling approach are integer values greater than zero.")
        raise ValueError(f"'num_processes' must be int >0, not {num_processes} for sub-sampling approach")


    if not type(warehouse_path) == str:
        logger.error(f"Encountered 'warehouse_path' parameter of type {type(warehouse_path).__name__}, while expecting type str.")
        raise TypeError(f"'warehouse_path' must be str, not {type(warehouse_path).__name__}")


def _check_dataset_filename(dataset_filename):
    """ Check if the dataset file exists and identify the data format from file extension """

    if os.path.isfile(dataset_filename):

        # identify the dataset filetype [ supported filetypes: 'csv', 'libsvm', 'arff' ]
        if '.' in dataset_filename:

            dataset_filetype = dataset_filename.split('.')[-1].lower()
            if dataset_filetype in ('csv', 'libsvm', 'arff'):
                return dataset_filetype

            else:
                logger.error(f"Unsupported data format: '{dataset_filetype}'. Supported data formats: 'csv', 'arff', 'libsvm'")
                raise ValueError(f"Unsupported data format: '{dataset_filetype}'")

        else:
            logger.error("Dataset filename doesn't have a file extension indicative of data format.\n"
                "Add appropriate file extension to dataset filename. Supported data formats: '.csv', '.arff', '.libsvm'")
            raise ValueError("No file extension for dataset file indicative of data format")

    else:
        logger.error(f"The specified dataset file '{dataset_filename}' could not be found.")
        raise FileNotFoundError("Specified dataset file not found")


def read_dataset_config_file(dataset_config_filename, dataset_filetype):
    """ Check if the dataset config file exists and validate its contents """

    if os.path.isfile(dataset_config_filename):
        spec = importlib.util.spec_from_file_location('dataset_config', dataset_config_filename)

        if spec is None:
            logger.error(f"Couldn't create `importlib.ModuleSpec` object for dataset configuration file '{dataset_config_filename}'. Ensure that the dataset configuration file has file extension '.py'.")
            raise ValueError("Couldn't create module spec for dataset configuration file")

        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"The dataset configuration file '{dataset_config_filename}' could not be executed: {str(e)}")
            raise ValueError(f"Dataset configuration file could not be executed: {str(e)}") from None

        try:
            dataset_config = getattr(module, 'config')
        except AttributeError:
            logger.error(f"The 'config' variable containing the dataset configuration object was not defined in the dataset configuration file '{dataset_config_filename}'.")
            raise ValueError("No 'config' variable in dataset config file") from None

        if type(dataset_config) == DATASET_FILETYPE_CONFIG[dataset_filetype]:
            return dataset_config

        else:
            logger.error(f"Invalid object type encountered for variable 'config' in dataset configuration file: expected: '{DATASET_FILETYPE_CONFIG[dataset_filetype]}', encountered: '{type(dataset_config)}'.")
            raise ValueError("Invalid type for variable 'config' in dataset config file")

    else:
        logger.error(f"The specified dataset configuration file '{dataset_config_filename}' could not be found.")
        raise FileNotFoundError("Dataset configuration file not found")


def _check_dataset(dataset):
    """ Check if the dataset is a binary classification dataset """

    # check that the dataset is a binary classification dataset
    if not ( hasattr(dataset, 'classes_') and dataset.classes_ is not None and len(dataset.classes_) == 2 ):
        logger.error("`automs` works only for binary (two class) classification problem. The specified dataset is " + ( "not a" if (not hasattr(dataset, 'classes_') or (dataset.classes_ is None) ) else f"a {len(dataset.classes_)} class") + " classification dataset.")
        raise ValueError("Dataset is not a binary classification dataset")


def create_current_run_warehouse_dir(warehouse_path, dataset_base_filename, is_user_configured_warehouse):
    """ Creates a directory in the warehouse for the current run of `automs` """

    if warehouse_path is not None:
        if os.path.isdir(warehouse_path):
            prefix = '' if is_user_configured_warehouse else 'automs-'
            suffix = ''
            current_run_warehouse_dirname = prefix + dataset_base_filename + suffix
            current_run_warehouse_path = os.path.join(warehouse_path, current_run_warehouse_dirname)

            if os.path.exists(current_run_warehouse_path):
                suffix_num = 1
                suffix = f'-{suffix_num}'
                current_run_warehouse_dirname = prefix + dataset_base_filename + suffix
                current_run_warehouse_path = os.path.join(warehouse_path, current_run_warehouse_dirname)
                while os.path.exists(current_run_warehouse_path):
                    suffix_num += 1
                    suffix = f'-{suffix_num}'
                    current_run_warehouse_dirname = prefix + dataset_base_filename + suffix
                    current_run_warehouse_path = os.path.join(warehouse_path, current_run_warehouse_dirname)

            try:
                os.mkdir(current_run_warehouse_path)
            except OSError as e:
                logger.error(f"Couldn't create directory for current run in specified warehouse: '{current_run_warehouse_path}'")
                raise ValueError("Couldn't create directory for current run in specified warehouse")
            else:
                return current_run_warehouse_path

        else:
            logger.error("The specified warehouse path '{warehouse_path}' doesn't exist.")
            raise ValueError("Specified warehouse path doesn't exist")

    else:
        logger.error("AutoMS warehouse not configured or specified. Either configure AutoMS by running `automs-config` in your terminal or specify a warehouse path.")
        raise ValueError("AutoMS warehouse not configured or specified")


def write_current_run_metainfo(current_run_warehouse_dir, dataset_filename, dataset_config, **kargs):
    """ Writes meta information about the current run of `automs` in the current run warehouse directory """

    current_run_metainfo_dict = OrderedDict(
        dataset_filename = os.path.abspath(dataset_filename),
        creation_timestamp =  str(datetime.now()),

        dataset_filetype = dataset_config.dataset_filetype,
        dataset_config = vars(dataset_config),

        **kargs)

    current_run_metainfo_filename = os.path.join(current_run_warehouse_dir, 'metainfo.json')
    with open(current_run_metainfo_filename, 'w') as current_run_metainfo_file:
        json.dump(current_run_metainfo_dict, current_run_metainfo_file, indent=4)


def read_bag_file(bag_filename):
    """ Read the subsampled dataset from the bag file """

    with open(bag_filename, 'rb') as bag_file:
        bag = pickle.load(bag_file)

    data, target = bag['data'], bag['target']

    dataset = EDA()
    dataset.load_data(data, target)

    return dataset


def extract_bag_index(bag_filename):
    """ Extract the index of bag corresponding to the bag filename """

    bag_filename = os.path.basename(bag_filename)
    bag_index = int(bag_filename.rstrip('.p').split('_')[-1].lstrip('bag'))

    return bag_index


def write_classification_complexity_to_file(clf_complexity, clf_complexity_filename):
    """ Write the classification complexity to specified file """

    with open(clf_complexity_filename, 'w') as f_clf_complexity:
        f_clf_complexity.write(f"IS HARD TO CLASSIFY = {clf_complexity}\n")


def write_f1_scores_to_file(clf_models_f1_scores, f1_scores_filename):
    """ Write the f1 scores to specified file """

    with open(f1_scores_filename, 'w') as f_f1_scores:
        for clf_model, f1_score in clf_models_f1_scores.items():
            f_f1_scores.write(f"{clf_model} = {f1_score}\n")


def write_consolidated_results_to_file(bags_cluster_indices, bags_predicted_clf_complexity, predicted_clf_complexity, clf_models_bags_estimated_f1_scores, clf_models_estimated_f1_scores, consolidated_results_filename):
    """ Write the consolidated results to spreadsheat file """

    flatten_bag_cluster_indices = lambda bag_cluster_indices: {(cluster_algorithm, cluster_indices_type, cluster_index_name) : cluster_index_value for cluster_algorithm, cluster_indices_types in bag_cluster_indices.items() for cluster_indices_type, cluster_indices_names in cluster_indices_types.items() for cluster_index_name, cluster_index_value in cluster_indices_names.items()}

    # Flatten the keys of multi-level 'cluster indices dictionary' into a flat dictionary for each bag in list
    bags_flattened_cluster_indices = list(map(flatten_bag_cluster_indices, bags_cluster_indices))

    # Aggregate the values for each key across bags into a list of values for each key
    flattened_bags_cluster_indices = {bag_flattened_cluster_indices_key: [ bag_flattened_cluster_indices[bag_flattened_cluster_indices_key] for bag_flattened_cluster_indices in bags_flattened_cluster_indices ] for bag_flattened_cluster_indices_key in bags_flattened_cluster_indices[0]}

    n_bags = len(bags_cluster_indices)
    bags_names = [f'Bag {i_bag}' for i_bag in range(1, n_bags+1)]
    consolidated_results_df = pd.DataFrame(flattened_bags_cluster_indices, index=bags_names)

    consolidated_results_df['Predicted Classification Complexity', 'Hardness Classifier Output', 'Is hard to classify?'] =  np.array(bags_predicted_clf_complexity, dtype=str)

    for clf_model, bags_clf_model_estimated_f1_scores in clf_models_bags_estimated_f1_scores.items():
        consolidated_results_df['Estimated F1 Scores', 'Classification Models', clf_model] = bags_clf_model_estimated_f1_scores

    # Assumption: 'clf_models_bags_estimated_f1_scores' and 'clf_models_estimated_f1_scores' have the same dict keys
    predicted_clf_complexity_series = pd.Series({('Predicted Classification Complexity', 'Hardness Classifier Output', 'Is hard to classify?') : str(predicted_clf_complexity) }, name='Dataset')
    estimated_f1_scores_clf_models_series = pd.Series({('Estimated F1 Scores', 'Classification Models', clf_model) : estimated_f1_score for clf_model, estimated_f1_score in clf_models_estimated_f1_scores.items()}, name='Dataset')
    dataset_outputs_series = pd.Series.append(predicted_clf_complexity_series, estimated_f1_scores_clf_models_series)

    consolidated_results_df = consolidated_results_df.append(dataset_outputs_series)

    consolidated_results_df.to_excel(consolidated_results_filename)

