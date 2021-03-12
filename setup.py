from setuptools import setup, find_packages


# Read the 'README.rst' for use as long description
def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name = "automs", # name of the package
    version = "0.1.0", # Version 0.1 Final Release

    description = "Automatic Model Selection Using Cluster Indices",
    long_description = readme(),
    long_description_content_type = "text/markdown",

    url = "https://automs.readthedocs.io", # Project homepage

    author = "Sudarsun Santhiappan, Nitin Shravan, Mukesh Reghu",
    author_email = "mail@sudarsun.in, ntnshrav@gmail.com, reghu.mukesh@gmail.com",

    maintainer = "Mukesh Reghu",
    maintainer_email = "reghu.mukesh@gmail.com",

    license = "3-clause-BSD",

    classifiers = [
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',

        'Operating System :: POSIX :: Linux',
    ],

    keywords = 'automatic classification model selection cluster indices',

    # When source code is an subdirectory under project root, e.g., 'src/', it is necessary to specify the `package_dir` argument
    package_dir = {'': 'src'},

    project_urls = {
        'Documentation': 'https://automs.readthedocs.io',
        'Source': 'https://github.com/RBC-DSAI-IITM/AutoMS',
        'Tracker': 'https://github.com/RBC-DSAI-IITM/AutoMS/issues',
        # could also add publication (abstract) url later
    },

    packages = find_packages(where='src'), # automatically find packages -- ['automs']

    # Refer to https://github.com/h5py/h5py/issues/535
    setup_requires = ['cython', 'numpy'],

    # TODO: Check if the code is compatible with the latest versions of these packages
    install_requires = [
        'npyscreen',

        'scipy',
        'matplotlib',
        'hdbscan',
        'pandas',
        'seaborn',
        'scikit_learn',
        'tqdm',
        'onnxruntime',
        'numpy',
        'xgboost',

        'cython',
        'openpyxl',
    ],

    entry_points = {
            'console_scripts': ['automs=automs.main:main']
    },

    package_data = {
        'automs': ['models/features.pkl',
                   'models/decision_tree_f1_estimator.onnx',
                   'models/random_forest_f1_estimator.onnx',
                   'models/logistic_regression_f1_estimator.onnx',
                   'models/k_nearest_neighbor_f1_estimator.onnx',
                   'models/xgboost_f1_estimator.onnx',
                   'models/support_vector_machine_f1_estimator.onnx',
                   'models/hardness_classifier.onnx'],
    },

    # Code was developed in python 3.6
    python_requires = '~=3.6',

    scripts = ['bin/automs-config'],
)
