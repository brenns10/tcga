#!/usr/bin/env python3
"""Contains menu for running various tasks in the TCGA package."""

from os.path import expandvars, expanduser

from tcga import parse
from smbio.util.menu import Menu, repeat_input


main = Menu(title='TCGA Procedures', reentrant=True)
state = {'muts': None, 'phen': None}


def check_save(data):
    """Ask if the user would like to save a CSV."""
    response = repeat_input('Save? ', str)
    if response.lower() not in ['y', 'yes', 'true', 't']:
        return
    location = repeat_input('Location: ', str)
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
