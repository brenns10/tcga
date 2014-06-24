#!/usr/bin/env python3
#-------------------------------------------------------------------------------
#
# File:         run.py
#
# Author:       Stephen Brennan
#
# Date Created: Tuesday, 24 June 2014
#
# Description:  Contains menu for running various tasks in the TCGA package.
#
#-------------------------------------------------------------------------------

from os.path import expandvars, expanduser

from tcga import parse, tree, util, compare

main = util.Menu(title='TCGA Procedures', reentrant=True)
state = {'muts': None, 'phen': None}


@main.add_function('Detection by Dataset Size')
def run_detection_by_size():
    """Run collect_detection_data."""
    res = compare.collect_detection_data()
    check_save(res)


@main.add_function('Detection by Data Distribution')
def run_detection_by_distribution():
    """Run detection_by_distribution."""
    res = compare.detection_by_distribution()
    check_save(res)


@main.add_function('Random Tree')
def run_random_tree():
    """Run the random tree algorithm."""
    check_data()
    depth = util.repeat_input(prompt='Depth: ', in_type=int)
    print('Beginning \'random_tree\'.')
    tree.random_tree(state['muts'], state['phen'], depth)


@main.add_function('Greedy Tree')
def run_greedy_tree():
    """Run the greedy tree algorithm."""
    check_data()
    print('Beginning \'greedy_tree\' ...')
    tree.greedy_tree(state['muts'], state['phen'])


def check_save(data):
    """Ask if the user would like to save a CSV."""
    response = util.repeat_input('Save? ', str)
    if response.lower() not in ['y', 'yes', 'true', 't']:
        return
    location = util.repeat_input('Location: ', str)
    location = expandvars(expanduser(location))
    data.to_csv(location)


def check_data():
    """Load the mutation data if it is not already loaded."""
    if state['muts'] is None or state['phen'] is None:
        reload_data()


@main.add_function('Reload Data')
def reload_data():
    """Force loading mutation data."""
    print('Loading data ...')
    state['muts'], state['phen'] = parse.data()


if __name__ == '__main__':
    main.display()