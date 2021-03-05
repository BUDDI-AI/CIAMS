""" Functions for reading, preprocessing and sampling datasets into bags """

# standard library imports
import logging

# local application code imports
from .config import DATASET_FILETYPE_CONFIG
from .utils import _check_dataset_filename
from .eda import EDA


# setup logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S',
            level = logging.INFO)
logger = logging.getLogger(__name__)


def read_dataset_file(dataset_filename, dataset_config):

    dataset_filetype = _check_dataset_filename(dataset_filename)

    # verify that the dataset file format inferred from dataset filename and the dataset config object type match
    if type(dataset_config) != DATASET_FILETYPE_CONFIG[dataset_filetype]:
        logger.error(f"Encountered dataset config object of type `{type(dataset_config)}` when expecting object of type `{DATASET_FILETYPE_CONFIG[dataset_filetype]}`")
        raise TypeError("Encountered invalid dataset config object")

    # Read the dataset into an :obj:`automs.eda.EDA` object
    dataset = EDA()

    if dataset_filetype == 'csv': dataset.read_data_csv(dataset_filename, **vars(dataset_config))
    elif dataset_filetype == 'libsvm': dataset.read_data_libsvm(dataset_filename, **vars(dataset_config))
    elif dataset_filetype == 'arff': dataset.read_data_arff(dataset_filename, **vars(dataset_config))
    else:
        logger.error(f"Specified dataset file's filetype or data format ('{dataset_filetype}') doesn't have an associated reader method in :class:`automs.eda.EDA`.")
        raise ValueError("No reader method for specified dataset's filetype")

    # Dummy code the nominal columns (or features)
    dataset.dummy_coding()

    return dataset


def sample_dataset(dataset, oneshot, sample_location):

    # Compute the number of bags and sample size for each bag
    if oneshot:
        sample_size = dataset.n_samples
        n_bags = 1

    else:
        if dataset.n_samples > 1000:
            sample_size = 500

        elif dataset.n_samples > 500:
            # sample size is half of dataset size
            sample_size = dataset.n_samples // 2

        else:
            logger.error(f"Dataset must have atlest 500 examples for sub-sampled bags setting. Dataset has only {dataset.n_samples} examples. Run the dataset in oneshot setting.")
            raise ValueError("Dataset too small for sub-sampled bags setting")

        n_bags = round(5 * dataset.n_samples / (0.63 * sample_size))

    logger.info(f"Number of bags = {n_bags}, Number of samples per bag = {sample_size}")

    stratified_sampling_results = dataset.random_stratified_sampling(sample_location, 'bags', sample_size, n_iterations=n_bags)

    return stratified_sampling_results
