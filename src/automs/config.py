""" Defines the dataset configuration class for csv, libsvm, arff data formats
Dataset configuration object contains information for reading and preprocessing the dataset file """

class CsvConfig:
    """ Dataset configuration class for CSV data format """

    dataset_filetype = 'csv'

    def __init__(self, sep=',', skiprows=None, header_row=None, usecols=None, target_col=-1, categorical_cols='infer', na_values=None, **kargs):
        """
        Parameters
        ----------

        sep : str, optional
            Column delimiter. **Accepted values:** ``None`` implies autodetect delimiter, ``'\s+'`` uses combination of spaces and tabs, regular expressions. (default is ``','`` ).

        skiprows : list of int or int, optional
            List of line indices to skip or the number of starting lines to skip. (default value ``None`` implies don't skip any lines)

        header_row : int, optional
            Relative Zero-Index (index of rows after skipping rows using ``skiprows`` parameter) of the row containing column names. Note: All preceding rows are ignored. (default value ``None`` implies no header row)

        usecols : list, optional
            List of column names (or column indices, if no header row specified) to consider. (default value  ``None`` indicates use of all columns)

        target_col : int, optional
            Relative Zero-Index of column (after filtering columns using ``usecols`` parameter) to use as target values. ``None`` indicates absence of target value columns. (default value ``-1`` implies use the last column as target values)

        categorical_cols : 'infer' or list or str or int or 'all', optional
            List (str or int if singleton) of column names (or absolute indices of columns, if no header row specified) of categorical columns to encode. Default value ``'infer'`` autodetects nominal categorical columns. ``'all'`` implies all columns are nominal categorical. ``None`` implies no nominal categorical columns exist.

        na_values : scalar or str or list-like or dict, optional
            Additional strings to recognize as NA/NaN. If dict is passed, it specifies per-column NA values. By default the following values are interpreted as NaN: ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’. (default value ``None`` implies no additional values to intrepret as NaN)

        **kargs
            Other keyword arguments accepted by :func:`pandas.read_csv` such as ``comment`` and ``lineterminator``.

        Notes
        -----
        * ``skiprows`` parameter uses absolute row indices whereas ``header_row`` parameter uses relative index (i.e., zero-index after removing rows specied by ``skiprows`` parameter).

        * ``usecols`` and ``categorical_cols`` parameters use absolute column names (or indices, if no header row) whereas ``target_cols`` parameter uses relative column indices (or names) after filtering out columns specified by ``usecols`` parameter.

        * ``categorical_cols='infer'`` identifies and encodes nominal features (i.e., features of 'string' type, with fewer unique entries than a value heuristically determined from the number of data samples) and drops other 'string' and 'date' type features from the dataset. Use :func:`automs.eda.max_classes_nominal` to find the heuristically determined value of maximum number of distinct entries in nominal features for a given number of samples.

        * Data samples with any NA/NaN features are implicitly dropped.
        """
        self.sep = sep
        self.skiprows = skiprows
        self.header_row = header_row
        self.usecols = usecols
        self.target_col = target_col
        self.encode_target = True
        self.categorical_cols = categorical_cols
        self.na_values = na_values
        self.nrows = None
        # self.kargs = kargs
        for key, value in kargs.items():
            setattr(self, key, value)


class LibsvmConfig:
    """ Dataset configuration class for LIBSVM data format """

    dataset_filetype = 'libsvm'

    def __init__(self):
        pass


class ArffConfig:
    """ Dataset configuration class for ARFF data format """

    dataset_filetype = 'arff'

    def __init__(self, target_attr='class', numeric_categorical_attrs=None):
        """
        Parameters
        ----------

        target_attr : str, optional
            Attribute name of the target column. ``None`` implies no target columns. (default value is ``'class'``)

        numeric_categorical_attrs : list of str, optional
            List of names of numeric attributes to be inferred as nominal and to be encoded. Note: All nominal attributes are implicitly encoded. (default value ``None`` implies no numeric attributes are to be infered as nominal)

        Notes
        -----
        All nominal type attributes are implicitly encoded.
        """
        self.target_attr = target_attr
        self.encode_target = True
        self.numeric_categorical_attrs = numeric_categorical_attrs


DATASET_FILETYPE_CONFIG = {
    'csv': CsvConfig,
    'libsvm': LibsvmConfig,
    'arff': ArffConfig
}
