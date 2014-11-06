#-------------------------------------------------------------------------------
# 
# File:         parse.py
#
# Author:       Stephen Brennan, Matthew Ruffalo
#
# Date Created: Tuesday,  3 June 2014
#
# Description:  Contains functions for parsing, saving, and loading data.
#
# This module operates on the idea of one central directory where all your
# input data is stored, and another central directory for your output data.
# The variables are IN_DIR and OUT_DIR, which are by default '~/data' and
# '~/results'.
#
# Furthermore, the idea of this module is that you should be able to
# reference your data by title, rather than typing out a filename.  So,
# if you have a Pickled data file named ~/data/mutations.pickle, you can
# simply load it by passing the title 'mutations', and the code will take it
# from there.
#
# Finally, this module makes the assumption that loading pickled data is
# better than parsing data.  So, it provides functions that access each type
# of data, first by looking for a pickled copy, and then by parsing it and
# pickling it, if the piclked copy did not exist.
#
#-------------------------------------------------------------------------------

import csv
from os.path import expanduser, expandvars, join, exists, isfile
import pickle

import numpy as np
import pandas as pd


IN_DIR = expanduser(expandvars('~/data'))
OUT_DIR = expanduser(expandvars('~/results'))


def _process_title(title, dir, ext=''):
    """
    Returns a filename from a title, directory, and optionally an extension.
    :param title: File title.
    :param dir: Directory file is in.
    :param ext: File extension (optional).
    :return: A full filename string.
    """
    # If the title contains user/variables, make sure to expand them
    title = expandvars(expanduser(title))
    # Create the full file path of the destination.
    return join(dir, title) + ext

def unpickle(title, dir=IN_DIR, ext='.pickle'):
    """
    Unpickles a Pickle file from the input data directory.
    :param title: The *title* of the data file *without extension*.  It
    should be a relative path from your data directory.
    :param dir: The directory this file is located within (defaults to DATA_DIR)
    :param ext: The file extension, defaults to '.pickle'
    :return: The unpickled file.
    """
    filename = _process_title(title, dir, ext)
    with open(filename, 'rb') as f:
        return pickle.load(f)


def to_pickle(obj, title, dir=OUT_DIR, ext='.pickle', allowoverwrite=False):
    """
    Pickles an object into a file with the given title.

    This is a convenience function for pickling data.  By default, it raises an
    error instead of overwriting data.  Also, it uses the default DATA_DIR,
    and appends a file extension of '.pickle', so you don't have to worry
    about filenames or locations.
    :param obj: Object to pickle.
    :param title: Title of target file (within the given directory, and with
    the extension appended).
    :param dir: Directory that the output file should be in (default OUT_DIR).
    :param ext: File extension (default '.pickle').
    :param allowoverwrite: Set this to true to suppress exceptions about
    overwriting existing files.
    :return: None
    """
    filename = _process_title(title, dir, ext)

    # Prevent overwriting (which Python doesn't seem to do by default).
    if exists(filename) and not allowoverwrite:
        raise FileExistsError('You are attempting to overwrite an existing '
                              'file.')

    # Finally, dump the file
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def _read_csv(filename):
    """
    Reads a CSV file, skipping header. Returns tuples containing each row.
    :param filename: The raw filename to read from.
    """
    with open(filename) as f:
        r = csv.reader(f)
        next(r)
        yield from r


def _create_pickle_caching_function(parse_function, default_title,
                                    default_dir, pass_title=True):
    """
    Generate a function that will load a data file by loading a pickled copy, or
    parsing and pickling it if unavailable.
    :param parse_function: The function used to parse the data file.  Needs
    to accept 'title' as first argument, and a keyword argument of 'dir'.
    :param default_title: The default title the generated function will use.
    :param default_dir: The default directory the generated function will
    look in.
    :param source_ext: The file extension of the source format (to be parsed).
    :return: The function described above.
    """
    def _pickle_caching_function(title=default_title, dir=default_dir,
                                 **kwargs):
        """
        Load a data file.  If it has a pickled version, load that.  If not,
        parse its original file, pickle it, and then return it.
        :param title: The title of the file (within dir).
        :param dir: The location of the file.
        :return: The data.
        """
        filename = _process_title(title, dir)

        # Return the pickled version if available.
        if isfile(filename + '.pickle'):
            return unpickle(title, dir=dir)

        # Parse and pickle to the location they loaded from.
        if pass_title:
            obj = parse_function(title, dir=dir, **kwargs)
        else:
            obj = parse_function(dir=dir, **kwargs)
        to_pickle(obj, title, dir=dir)
        return obj
    return _pickle_caching_function


def parse_mutations(title='mutations', dir=IN_DIR, ext='.csv'):
    """
    Parse the mutation file from CSV, and return it as a pandas DataFrame.
    :param title: Title of the file, defaults to 'mutations'.
    :param dir: Directory of file, defaults to IN_DIR.
    :param ext: Extension of file, defaults to '.csv'
    :return: The mutation table as pandas DataFrame.
    """
    filename = _process_title(title, dir, ext)

    # Get a list of patient, gene tuples from the file.
    raw_muts = list(_read_csv(filename))
    # Convert to a list of genes and a list of patients.
    genes, patients = zip(*raw_muts)  # unzip the zipped lists
    # Remove duplicates in the genes and patients
    genes = set(genes)
    patients = set(patients)

    # Create a dense mutation DataFrame with the patients as rows and the genes
    # as columns.
    muts = pd.DataFrame(np.zeros((len(patients), len(genes))),
                        index=patients, columns=genes, dtype=bool)
    # Populate the DataFrame with the list of tuples.
    for gene, patient in raw_muts:
        muts.ix[patient, gene] = 1
    return muts


# Create the utility function 'mutations'
mutations = _create_pickle_caching_function(parse_mutations, 'mutations',
                                            IN_DIR)


def parse_binary_phenotypes(title='vital-status', dir=IN_DIR, ext='.csv',
                            dtype=bool):
    """
    Parse a binary phenotype CSV and return it as a pandas Series.
    :param title: The title of the file (within dir), default='vital-status'.
    :param dir: The directory of the file, default=IN_DIR.
    :param ext: The file extension, default = '.csv'.
    :param dtype: The data type to have pandas coerce the values to.
    :return: A pandas Series containing the CSV data.
    """
    filename = _process_title(title, dir, ext)
    val = pd.read_csv(filename, index_col=0, squeeze=True)
    if dtype:
        return val.astype(dtype)
    else:
        return val


def parse_lifetime_phenotypes(title='lifetime', dir=IN_DIR, ext='.csv'):
    """
    Parse a lifetime phenotype CSV and return it as a pandas DataFrame.
    :param title: Title of the file
    :param dir: Directory (default IN_DIR).
    :param ext: Extension (csv)
    :return: Lifetime DataFrame
    """
    filename = _process_title(title, dir, ext)
    df = pd.read_csv(filename, index_col=0, squeeze=True)
    df = df.convert_objects(convert_numeric=True)
    return df


def _get_phenotype_parser(filename):
    """
    Return the correct parser function for a phenotype file.  Reads the CSV
    and uses the number of fields to determine.
    :param filename: Filename
    :return: Parser function
    """
    with open(filename) as f:
        r = csv.reader(f)
        header = next(r)
        if len(header) == 2:
            return parse_binary_phenotypes
        elif len(header) == 3:
            return parse_lifetime_phenotypes


def phenotypes(title='vital-status', dir=IN_DIR, ext='.csv', **kwargs):
    """
    Get phenotype data with the given title.  Cache as pickle for future use.
    Determines what type of phenotype by the contents of the file.
    :param title: Title (i.e. filename w/o extension or directory)
    :param dir: Directory (default to input directory)
    :param ext: Extension (default to csv)
    :param kwargs: Arguments to pass to the parser function
    :return: The phenotype data file.
    """
    filename = _process_title(title, dir, ext)
    pickle_name = _process_title(title, dir, '.pickle')

    if isfile(pickle_name):
        return unpickle(title, dir)

    parser = _get_phenotype_parser(filename)
    data = parser(title, dir, ext, **kwargs)
    to_pickle(data, title, dir=dir)
    return data


def reindex(muts, phen):
    """
    Handle differing indices between the mutation and phenotype data.

    Make sure we only use the patients for which we have both mutation data
    and phenotype data.

    :param muts: The mutation dataset (pandas DataFrame)
    :param phen: The phenotype dataset (pandas Series)
    :return: a 2-tuple:
    [0]: modified mutations dataframe
    [1]: modified phenotypes dataframe
    """
    new_index = muts.index.intersection(phen.index)
    muts = muts.reindex(index=new_index)
    phen = phen.reindex(index=new_index)
    return muts, phen


def data(mutation_title='mutations', phenotype_title='vital-status',
         dir=IN_DIR, do_reindex=True):
    """
    Read in a mutation file and a phenotype file.

    By default, this function calls reindex() on the datasets so that only
    patients which occur in the phenotype are included in calculations.
    :param mutation_title: The title of the mutation file, default='mutations'
    :param phenotype_title: The title of the phenotype file,
    default='vital-status'.
    :param dir: The directory to look for input files, by default IN_DIR.
    :param reindex: Whether or not to reindex.  By default, True.
    :return: A 2-tuple:
    [0]: Mutation DataFrame
    [1]: Phenotype DataFrame
    """
    muts = mutations(title=mutation_title, dir=dir)
    phen = phenotypes(title=phenotype_title, dir=dir)
    if do_reindex:
        muts, phen = reindex(muts, phen)
    return muts, phen


def parse_sparse_mutations(title='mutations', dir=IN_DIR, ext='.csv'):
    """
    Reads the sparse mutations, and returns a list of (gene, patient) tuples.
    Restricts to only the patients included in the phenotype index,
    if phenotype is provided.
    :param filename: The filename of the mutations csv file.
    :param phenotype: The phenotype Series.
    :return: A Python list of (gene, patient) tuples.
    """
    filename = _process_title(title, dir, ext)
    muts = list(_read_csv(filename))
    return list(set(muts))  # no duplicates


sparse_mutations = _create_pickle_caching_function(parse_sparse_mutations,
                                                   'sparse-mutations',
                                                   IN_DIR, False)


def restrict_sparse_mutations(spmuts, phenotype, mutations):
    """
    Restrict the sparse mutations to only patients that exist in the
    phenotype data.
    :param spmuts: Sparse mutation dataset.
    :param phenotype: Phenotype dataset.
    :return: New sparse mutation list.
    """
    return [(gene, patient) for gene, patient in spmuts if patient in
            phenotype.index and gene in mutations]


def dag(title='dag-nx-relabeled', dir=IN_DIR):
    """
    Unpickle the DAG.  This is really just an alias for unpickle().
    :param title: Title of dag file.
    :param dir: Directory.
    :return: Unpickled dag.
    """
    return unpickle(title, dir)