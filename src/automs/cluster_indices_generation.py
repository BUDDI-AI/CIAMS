""" Module containing functions for clustering and cluster indices generation from samples of data """
import logging

import numpy as np
import tqdm

from .utils import read_bag_file, extract_bag_index
from .internal_indices import InternalIndices, INTERNAL_INDICES_METHOD_NAMES_DICT
from .external_indices import ExternalIndices, EXTERNAL_INDICES_METHOD_NAMES_DICT


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
			datefmt = '%m/%d/%Y %H:%M:%S',
			level = logging.INFO)
logger = logging.getLogger(__name__)


KMEANS_CHOOSEN_CLUSTER_INDICES = {

    'internal_indices' : [
                            # 'WGSS',
                            # 'BGSS',
                            # 'Ball-Hall',
                            # 'Banfeld-Raftery',
                            # 'Calinski-Harabasz',
                            # 'Det-Ratio',
                            # 'Ksq-DetW',
                            # 'Log-Det-Ratio',
                            # 'Log-SS-Ratio',
                            ## 'Scott-Symons',
                            'Silhouette',
                            # 'Trace-WiB',
                            'C',
                            'Dunn',
                            # 'Davies-Bouldin',
                            # 'Ray-Turi',
                            # 'PBM',
                            'Score',
                         ],

    'external_indices' : [
                            'Entropy',
                            'Purity',
                            'Precision',
                            'Recall',
                            'F',
                            'Weighted-F',
                            'Folkes-Mallows',
                            'Rand',
                            'Adjusted-Rand',
                            'Adjusted-Mutual-Info',
                            'Normalized-Mutual-Info',
                            'Homogeneity',
                            'Completeness',
                            'V-Measure',
                            'Jaccard',
                            'Hubert Γ̂',
                            'Kulczynski',
                            # 'McNemar',
                            'Phi',
                            'Russel-Rao',
                            'Rogers-Tanimoto',
                            'Sokal-Sneath1',
                            'Sokal-Sneath2',
                         ],

}


HIERARCHICAL_CHOOSEN_CLUSTER_INDICES = {

    'internal_indices' : [
                            # 'WGSS',
                            # 'BGSS',
                            # 'Ball-Hall',
                            # 'Banfeld-Raftery',
                            # 'Calinski-Harabasz',
                            # 'Det-Ratio',
                            # 'Ksq-DetW',
                            # 'Log-Det-Ratio',
                            # 'Log-SS-Ratio',
                            ## 'Scott-Symons',
                            'Silhouette',
                            # 'Trace-WiB',
                            'C',
                            'Dunn',
                            # 'Davies-Bouldin',
                            # 'Ray-Turi',
                            # 'PBM',
                            'Score',
                         ],

    'external_indices' : [
                            'Entropy',
                            'Purity',
                            'Precision',
                            'Recall',
                            'F',
                            'Weighted-F',
                            'Folkes-Mallows',
                            'Rand',
                            'Adjusted-Rand',
                            'Adjusted-Mutual-Info',
                            'Normalized-Mutual-Info',
                            'Homogeneity',
                            'Completeness',
                            'V-Measure',
                            'Jaccard',
                            'Hubert Γ̂',
                            'Kulczynski',
                            # 'McNemar',
                            'Phi',
                            'Russel-Rao',
                            'Rogers-Tanimoto',
                            'Sokal-Sneath1',
                            'Sokal-Sneath2',
                         ],

}


SPECTRAL_CHOOSEN_CLUSTER_INDICES = {

    'internal_indices' : [
                            # 'WGSS',
                            # 'BGSS',
                            # 'Ball-Hall',
                            # 'Banfeld-Raftery',
                            # 'Calinski-Harabasz',
                            # 'Det-Ratio',
                            # 'Ksq-DetW',
                            # 'Log-Det-Ratio',
                            # 'Log-SS-Ratio',
                            ## 'Scott-Symons',
                            'Silhouette',
                            # 'Trace-WiB',
                            'C',
                            'Dunn',
                            'Davies-Bouldin',
                            'Ray-Turi',
                            # 'PBM',
                            'Score',
                         ],

    'external_indices' : [
                            'Entropy',
                            'Purity',
                            'Precision',
                            'Recall',
                            'F',
                            'Weighted-F',
                            'Folkes-Mallows',
                            'Rand',
                            'Adjusted-Rand',
                            'Adjusted-Mutual-Info',
                            'Normalized-Mutual-Info',
                            'Homogeneity',
                            'Completeness',
                            'V-Measure',
                            'Jaccard',
                            'Hubert Γ̂',
                            'Kulczynski',
                            # 'McNemar',
                            'Phi',
                            'Russel-Rao',
                            'Rogers-Tanimoto',
                            'Sokal-Sneath1',
                            'Sokal-Sneath2',
                         ],

}


HDBSCAN_CHOOSEN_CLUSTER_INDICES = {

    'internal_indices' : [
                            # 'WGSS',
                            # 'BGSS',
                            # 'Ball-Hall',
                            # 'Banfeld-Raftery',
                            # 'Calinski-Harabasz',
                            # 'Det-Ratio',
                            # 'Ksq-DetW',
                            # 'Log-Det-Ratio',
                            'Log-SS-Ratio',
                            ## 'Scott-Symons',
                            'Silhouette',
                            # 'Trace-WiB',
                            # 'C',
                            'Dunn',
                            # 'Davies-Bouldin',
                            # 'Ray-Turi',
                            # 'PBM',
                            'Score',
                         ],

    'external_indices' : [
                            'Entropy',
                            'Purity',
                            'Precision',
                            'Recall',
                            'F',
                            'Weighted-F',
                            'Folkes-Mallows',
                            'Rand',
                            'Adjusted-Rand',
                            'Adjusted-Mutual-Info',
                            'Normalized-Mutual-Info',
                            'Homogeneity',
                            'Completeness',
                            'V-Measure',
                            'Jaccard',
                            'Hubert Γ̂',
                            'Kulczynski',
                            # 'McNemar',
                            'Phi',
                            'Russel-Rao',
                            'Rogers-Tanimoto',
                            'Sokal-Sneath1',
                            'Sokal-Sneath2',
                         ],

}


FEATURE_VECTOR_CLUSTER_INDICES_ORDER_TRIPLES = [

    # ('kmeans', 'internal_indices', 'WGSS'),
    # ('kmeans', 'internal_indices', 'BGSS'),
    # ('kmeans', 'internal_indices', 'Ball-Hall'),
    # ('kmeans', 'internal_indices', 'Banfeld-Raftery'),
    # ('kmeans', 'internal_indices', 'Calinski-Harabasz'),
    # ('kmeans', 'internal_indices', 'Det-Ratio'),
    # ('kmeans', 'internal_indices', 'Ksq-DetW'),
    # ('kmeans', 'internal_indices', 'Log-Det-Ratio'),
    # ('kmeans', 'internal_indices', 'Log-SS-Ratio'),
    ## ('kmeans', 'internal_indices', 'Scott-Symons'),
    ('kmeans', 'internal_indices', 'Silhouette'),
    # ('kmeans', 'internal_indices', 'Trace-WiB'),
    ('kmeans', 'internal_indices', 'C'),
    ('kmeans', 'internal_indices', 'Dunn'),
    # ('kmeans', 'internal_indices', 'Davies-Bouldin'),
    # ('kmeans', 'internal_indices', 'Ray-Turi'),
    # ('kmeans', 'internal_indices', 'PBM'),
    ('kmeans', 'internal_indices', 'Score'),

    ('kmeans', 'external_indices', 'Entropy'),
    ('kmeans', 'external_indices', 'Purity'),
    ('kmeans', 'external_indices', 'Precision'),
    ('kmeans', 'external_indices', 'Recall'),
    ('kmeans', 'external_indices', 'F'),
    ('kmeans', 'external_indices', 'Weighted-F'),
    ('kmeans', 'external_indices', 'Folkes-Mallows'),
    ('kmeans', 'external_indices', 'Rand'),
    ('kmeans', 'external_indices', 'Adjusted-Rand'),
    ('kmeans', 'external_indices', 'Adjusted-Mutual-Info'),
    ('kmeans', 'external_indices', 'Normalized-Mutual-Info'),
    ('kmeans', 'external_indices', 'Homogeneity'),
    ('kmeans', 'external_indices', 'Completeness'),
    ('kmeans', 'external_indices', 'V-Measure'),
    ('kmeans', 'external_indices', 'Jaccard'),
    ('kmeans', 'external_indices', 'Hubert Γ̂'),
    ('kmeans', 'external_indices', 'Kulczynski'),
    # ('kmeans', 'external_indices', 'McNemar'),
    ('kmeans', 'external_indices', 'Phi'),
    ('kmeans', 'external_indices', 'Russel-Rao'),
    ('kmeans', 'external_indices', 'Rogers-Tanimoto'),
    ('kmeans', 'external_indices', 'Sokal-Sneath1'),
    ('kmeans', 'external_indices', 'Sokal-Sneath2'),


    # ('hierarchical', 'internal_indices', 'WGSS'),
    # ('hierarchical', 'internal_indices', 'BGSS'),
    # ('hierarchical', 'internal_indices', 'Ball-Hall'),
    # ('hierarchical', 'internal_indices', 'Banfeld-Raftery'),
    # ('hierarchical', 'internal_indices', 'Calinski-Harabasz'),
    # ('hierarchical', 'internal_indices', 'Det-Ratio'),
    # ('hierarchical', 'internal_indices', 'Ksq-DetW'),
    # ('hierarchical', 'internal_indices', 'Log-Det-Ratio'),
    # ('hierarchical', 'internal_indices', 'Log-SS-Ratio'),
    ## ('hierarchical', 'internal_indices', 'Scott-Symons'),
    ('hierarchical', 'internal_indices', 'Silhouette'),
    # ('hierarchical', 'internal_indices', 'Trace-WiB'),
    ('hierarchical', 'internal_indices', 'C'),
    ('hierarchical', 'internal_indices', 'Dunn'),
    # ('hierarchical', 'internal_indices', 'Davies-Bouldin'),
    # ('hierarchical', 'internal_indices', 'Ray-Turi'),
    # ('hierarchical', 'internal_indices', 'PBM'),
    ('hierarchical', 'internal_indices', 'Score'),

    ('hierarchical', 'external_indices', 'Entropy'),
    ('hierarchical', 'external_indices', 'Purity'),
    ('hierarchical', 'external_indices', 'Precision'),
    ('hierarchical', 'external_indices', 'Recall'),
    ('hierarchical', 'external_indices', 'F'),
    ('hierarchical', 'external_indices', 'Weighted-F'),
    ('hierarchical', 'external_indices', 'Folkes-Mallows'),
    ('hierarchical', 'external_indices', 'Rand'),
    ('hierarchical', 'external_indices', 'Adjusted-Rand'),
    ('hierarchical', 'external_indices', 'Adjusted-Mutual-Info'),
    ('hierarchical', 'external_indices', 'Normalized-Mutual-Info'),
    ('hierarchical', 'external_indices', 'Homogeneity'),
    ('hierarchical', 'external_indices', 'Completeness'),
    ('hierarchical', 'external_indices', 'V-Measure'),
    ('hierarchical', 'external_indices', 'Jaccard'),
    ('hierarchical', 'external_indices', 'Hubert Γ̂'),
    ('hierarchical', 'external_indices', 'Kulczynski'),
    # ('hierarchical', 'external_indices', 'McNemar'),
    ('hierarchical', 'external_indices', 'Phi'),
    ('hierarchical', 'external_indices', 'Russel-Rao'),
    ('hierarchical', 'external_indices', 'Rogers-Tanimoto'),
    ('hierarchical', 'external_indices', 'Sokal-Sneath1'),
    ('hierarchical', 'external_indices', 'Sokal-Sneath2'),


    # ('spectral', 'internal_indices', 'WGSS'),
    # ('spectral', 'internal_indices', 'BGSS'),
    # ('spectral', 'internal_indices', 'Ball-Hall'),
    # ('spectral', 'internal_indices', 'Banfeld-Raftery'),
    # ('spectral', 'internal_indices', 'Calinski-Harabasz'),
    # ('spectral', 'internal_indices', 'Det-Ratio'),
    # ('spectral', 'internal_indices', 'Ksq-DetW'),
    # ('spectral', 'internal_indices', 'Log-Det-Ratio'),
    # ('spectral', 'internal_indices', 'Log-SS-Ratio'),
    ## ('spectral', 'internal_indices', 'Scott-Symons'),
    ('spectral', 'internal_indices', 'Silhouette'),
    # ('spectral', 'internal_indices', 'Trace-WiB'),
    ('spectral', 'internal_indices', 'C'),
    ('spectral', 'internal_indices', 'Dunn'),
    ('spectral', 'internal_indices', 'Davies-Bouldin'),
    ('spectral', 'internal_indices', 'Ray-Turi'),
    # ('spectral', 'internal_indices', 'PBM'),
    ('spectral', 'internal_indices', 'Score'),

    ('spectral', 'external_indices', 'Entropy'),
    ('spectral', 'external_indices', 'Purity'),
    ('spectral', 'external_indices', 'Precision'),
    ('spectral', 'external_indices', 'Recall'),
    ('spectral', 'external_indices', 'F'),
    ('spectral', 'external_indices', 'Weighted-F'),
    ('spectral', 'external_indices', 'Folkes-Mallows'),
    ('spectral', 'external_indices', 'Rand'),
    ('spectral', 'external_indices', 'Adjusted-Rand'),
    ('spectral', 'external_indices', 'Adjusted-Mutual-Info'),
    ('spectral', 'external_indices', 'Normalized-Mutual-Info'),
    ('spectral', 'external_indices', 'Homogeneity'),
    ('spectral', 'external_indices', 'Completeness'),
    ('spectral', 'external_indices', 'V-Measure'),
    ('spectral', 'external_indices', 'Jaccard'),
    ('spectral', 'external_indices', 'Hubert Γ̂'),
    ('spectral', 'external_indices', 'Kulczynski'),
    # ('spectral', 'external_indices', 'McNemar'),
    ('spectral', 'external_indices', 'Phi'),
    ('spectral', 'external_indices', 'Russel-Rao'),
    ('spectral', 'external_indices', 'Rogers-Tanimoto'),
    ('spectral', 'external_indices', 'Sokal-Sneath1'),
    ('spectral', 'external_indices', 'Sokal-Sneath2'),


    # ('hdbscan', 'internal_indices', 'WGSS'),
    # ('hdbscan', 'internal_indices', 'BGSS'),
    # ('hdbscan', 'internal_indices', 'Ball-Hall'),
    # ('hdbscan', 'internal_indices', 'Banfeld-Raftery'),
    # ('hdbscan', 'internal_indices', 'Calinski-Harabasz'),
    # ('hdbscan', 'internal_indices', 'Det-Ratio'),
    # ('hdbscan', 'internal_indices', 'Ksq-DetW'),
    # ('hdbscan', 'internal_indices', 'Log-Det-Ratio'),
    ('hdbscan', 'internal_indices', 'Log-SS-Ratio'),
    ## ('hdbscan', 'internal_indices', 'Scott-Symons'),
    ('hdbscan', 'internal_indices', 'Silhouette'),
    # ('hdbscan', 'internal_indices', 'Trace-WiB'),
    # ('hdbscan', 'internal_indices', 'C'),
    ('hdbscan', 'internal_indices', 'Dunn'),
    # ('hdbscan', 'internal_indices', 'Davies-Bouldin'),
    # ('hdbscan', 'internal_indices', 'Ray-Turi'),
    # ('hdbscan', 'internal_indices', 'PBM'),
    ('hdbscan', 'internal_indices', 'Score'),

    ('hdbscan', 'external_indices', 'Entropy'),
    ('hdbscan', 'external_indices', 'Purity'),
    ('hdbscan', 'external_indices', 'Precision'),
    ('hdbscan', 'external_indices', 'Recall'),
    ('hdbscan', 'external_indices', 'F'),
    ('hdbscan', 'external_indices', 'Weighted-F'),
    ('hdbscan', 'external_indices', 'Folkes-Mallows'),
    ('hdbscan', 'external_indices', 'Rand'),
    ('hdbscan', 'external_indices', 'Adjusted-Rand'),
    ('hdbscan', 'external_indices', 'Adjusted-Mutual-Info'),
    ('hdbscan', 'external_indices', 'Normalized-Mutual-Info'),
    ('hdbscan', 'external_indices', 'Homogeneity'),
    ('hdbscan', 'external_indices', 'Completeness'),
    ('hdbscan', 'external_indices', 'V-Measure'),
    ('hdbscan', 'external_indices', 'Jaccard'),
    ('hdbscan', 'external_indices', 'Hubert Γ̂'),
    ('hdbscan', 'external_indices', 'Kulczynski'),
    # ('hdbscan', 'external_indices', 'McNemar'),
    ('hdbscan', 'external_indices', 'Phi'),
    ('hdbscan', 'external_indices', 'Russel-Rao'),
    ('hdbscan', 'external_indices', 'Rogers-Tanimoto'),
    ('hdbscan', 'external_indices', 'Sokal-Sneath1'),
    ('hdbscan', 'external_indices', 'Sokal-Sneath2'),

]


def generate_kmeans_cluster_indices(dataset, choosen_indices, n_jobs=None):

    cluster_labels = dataset.perform_kmeans_clustering(n_clusters='n_classes', n_jobs=n_jobs)

    internal_indices_values = dict()
    internal_validation = InternalIndices(dataset.data, cluster_labels)

    choosen_internal_indices = choosen_indices['internal_indices']
    for internal_index in choosen_internal_indices:
        internal_index_method = getattr(internal_validation, INTERNAL_INDICES_METHOD_NAMES_DICT[internal_index])
        internal_indices_values[internal_index] = internal_index_method()

    external_indices_values = dict()
    external_validation = ExternalIndices(dataset.target, cluster_labels)

    choosen_external_indices = choosen_indices['external_indices']
    for external_index in choosen_external_indices:
        external_index_method = getattr(external_validation, EXTERNAL_INDICES_METHOD_NAMES_DICT[external_index])
        external_indices_values[external_index] = external_index_method()

    indices_values = {
        'internal_indices' : internal_indices_values,
        'external_indices' : external_indices_values,
    }

    return indices_values


def generate_hierarchical_cluster_indices(dataset, choosen_indices, n_jobs=None):

    cluster_labels = dataset.perform_hierarchical_clustering(n_clusters='n_classes')

    internal_indices_values = dict()
    internal_validation = InternalIndices(dataset.data, cluster_labels)

    choosen_internal_indices = choosen_indices['internal_indices']
    for internal_index in choosen_internal_indices:
        internal_index_method = getattr(internal_validation, INTERNAL_INDICES_METHOD_NAMES_DICT[internal_index])
        internal_indices_values[internal_index] = internal_index_method()

    external_indices_values = dict()
    external_validation = ExternalIndices(dataset.target, cluster_labels)

    choosen_external_indices = choosen_indices['external_indices']
    for external_index in choosen_external_indices:
        external_index_method = getattr(external_validation, EXTERNAL_INDICES_METHOD_NAMES_DICT[external_index])
        external_indices_values[external_index] = external_index_method()

    indices_values = {
        'internal_indices' : internal_indices_values,
        'external_indices' : external_indices_values,
    }

    return indices_values


def generate_spectral_cluster_indices(dataset, choosen_indices, n_jobs=None):

    cluster_labels = dataset.perform_spectral_clustering(n_clusters='n_classes', n_jobs=n_jobs)

    internal_indices_values = dict()
    internal_validation = InternalIndices(dataset.data, cluster_labels)

    choosen_internal_indices = choosen_indices['internal_indices']
    for internal_index in choosen_internal_indices:
        internal_index_method = getattr(internal_validation, INTERNAL_INDICES_METHOD_NAMES_DICT[internal_index])
        internal_indices_values[internal_index] = internal_index_method()

    external_indices_values = dict()
    external_validation = ExternalIndices(dataset.target, cluster_labels)

    choosen_external_indices = choosen_indices['external_indices']
    for external_index in choosen_external_indices:
        external_index_method = getattr(external_validation, EXTERNAL_INDICES_METHOD_NAMES_DICT[external_index])
        external_indices_values[external_index] = external_index_method()

    indices_values = {
        'internal_indices' : internal_indices_values,
        'external_indices' : external_indices_values,
    }

    return indices_values


def generate_hdbscan_cluster_indices(dataset, choosen_indices, n_jobs=None):

    cluster_labels = dataset.perform_hdbscan_clustering(core_dist_n_jobs=(n_jobs if n_jobs is not None else 4))

    internal_indices_values = dict()
    internal_validation = InternalIndices(dataset.data, cluster_labels)

    choosen_internal_indices = choosen_indices['internal_indices']
    for internal_index in choosen_internal_indices:
        internal_index_method = getattr(internal_validation, INTERNAL_INDICES_METHOD_NAMES_DICT[internal_index])
        internal_indices_values[internal_index] = internal_index_method()

    external_indices_values = dict()
    external_validation = ExternalIndices(dataset.target, cluster_labels)

    choosen_external_indices = choosen_indices['external_indices']
    for external_index in choosen_external_indices:
        external_index_method = getattr(external_validation, EXTERNAL_INDICES_METHOD_NAMES_DICT[external_index])
        external_indices_values[external_index] = external_index_method()

    indices_values = {
        'internal_indices' : internal_indices_values,
        'external_indices' : external_indices_values,
    }

    return indices_values


def bag_generate_cluster_indices(bag_filename, n_jobs=1):
    """ Perform clustering on the bags and generate cluster indices evaluating the quality of clusters """

    bag_index = extract_bag_index(bag_filename)
    dataset = read_bag_file(bag_filename)

    logger.info("Bag %2d : performing kmeans clustering, generating cluster indices", bag_index)
    kmeans_cluster_indices = generate_kmeans_cluster_indices(dataset, KMEANS_CHOOSEN_CLUSTER_INDICES, n_jobs=n_jobs)
    logger.info("Bag %2d : performing hierarchical clustering, generating cluster indices", bag_index)
    hierarchical_cluster_indices = generate_hierarchical_cluster_indices(dataset, HIERARCHICAL_CHOOSEN_CLUSTER_INDICES, n_jobs=n_jobs)
    logger.info("Bag %2d : performing spectral clustering, generating cluster indices", bag_index)
    spectral_cluster_indices = generate_spectral_cluster_indices(dataset, SPECTRAL_CHOOSEN_CLUSTER_INDICES, n_jobs=n_jobs)
    logger.info("Bag %2d : performing hdbscan clustering, generating cluster indices", bag_index)
    hdbscan_cluster_indices = generate_hdbscan_cluster_indices(dataset, HDBSCAN_CHOOSEN_CLUSTER_INDICES, n_jobs=n_jobs)

    cluster_indices = {
            'kmeans' : kmeans_cluster_indices,
            'hierarchical' : hierarchical_cluster_indices,
            'spectral' : spectral_cluster_indices,
            'hdbscan' : hdbscan_cluster_indices,
    }

    return cluster_indices


def convert_cluster_indices_to_features(cluster_indices):
    """ Convert the cluster indices into a flat feature vector """

    feature_vector = list(map(
            lambda keys_triple: cluster_indices[keys_triple[0]][keys_triple[1]][keys_triple[2]],
            FEATURE_VECTOR_CLUSTER_INDICES_ORDER_TRIPLES
        ))

    feature_vector = np.array(feature_vector)

    return feature_vector
