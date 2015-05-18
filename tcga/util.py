"""Contains useful functions and classes."""

import traceback
from types import FunctionType
from io import StringIO
from enum import Enum
import sys


def dataframe_append(dataframe, rowdict):
    """
    Shortcut method for appending a row to a DataFrame.
    :param dataframe: The DataFrame to append to.
    :param rowdict: A dictionary containing each column's value.
    """
    newrow = len(dataframe)
    dataframe.loc[newrow] = 0  # init with 0's
    for k, v in rowdict.items():
        dataframe.loc[newrow, k] = v


class TermType(Enum):
    """Enumeration of types of terminals."""
    TTY = 1
    IPythonTerminal = 2
    IPythonGUI = 3
    File = 4
    Unknown = 0


def _non_ipy_term_type():
    """
    The terminal type of a non-IPython terminal.
    :return: Item of type TermType.
    """
    import sys
    if sys.stdout.isatty():
        return TermType.TTY
    else:
        return TermType.File


def get_term_type():
    """
    Identifies the type of terminal the current Python instance is running in.
    :return: Item of type TermType.
    """
    try:
        # noinspection PyUnresolvedReferences
        import IPython
    except ImportError:
        return _non_ipy_term_type()

    import IPython.terminal.interactiveshell as intshell
    import IPython.kernel.zmq.zmqshell as zmqshell
    ipy = IPython.get_ipython()
    if ipy is None:
        return _non_ipy_term_type()
    elif isinstance(ipy, intshell.TerminalInteractiveShell):
        return TermType.IPythonTerminal
    elif isinstance(ipy, zmqshell.ZMQInteractiveShell):
        return TermType.IPythonGUI
    else:
        return TermType.Unknown


def _silent_format(string, params):
    """
    Attempt to format a string, and ignore any exceptions that occur.
    :param string: String to format.
    :param params: Formatting parameters.
    :return: The formatted string, or the string parameter on error.
    """
    try:
        return string % params
    except TypeError:  # Not all arguments converted ...
        return string


class Progress:
    """
    An iterator which draws a progress bar on stdout.

    This iterator allows a progress bar to be drawn on stdout, while still
    allowing user code to print to stdout.  Note that the current
    implementation replaces stdout with a buffer and prints the buffer every
    time next() is called on the iterator.  Therefore, a progress bar within
    a progress bar would be hopelessly pointless, as it would never display
    properly.
    """

    def __init__(self, it, width=80, niters=100):
        """
        Create a progress bar from an iterator.

        This function can infer iterations from any object on which len() can
        be applied.  If len() cannot be applied, then niters is used to
        determine the number of iterations to base the estimate.

        :param it: The iterator to wrap.
        :param width: The console width.
        :param niters: Estimated number of iterations.
        :return:
        """
        self.it = iter(it)
        self.width = width
        self.iters = 0
        self.percent = 0
        self.needswrite = True
        self.finalized = False

        # Redirect stdout (mucho dangerous, I know)
        self.stdout = sys.stdout
        sys.stdout = StringIO()

        # Default formats
        self.prefix = '%3d%% ['
        self.suffix = ']'
        self.block = '#'

        # Attempt to figure out the amount of iterations:
        try:
            self.estimate = len(it)
        except TypeError:
            if niters is not None:
                self.estimate = niters
            else:
                self.estimate = 100

        self.__progress()

    def __iter__(self):
        """
        Called to create an iterator from this object.
        :return: Self.
        """
        return self

    def __flush(self):
        """
        Flush the stdout buffer, if it contains anything.
        :return: Nothing
        """
        output = sys.stdout.getvalue()
        if len(output) > 0:
            self.stdout.write('\r' + ' '*self.width + '\r')
            self.stdout.write(output)
            sys.stdout.close()
            sys.stdout = StringIO()
            self.needswrite = True

    def __progress(self):
        """
        Print the progress bar if it is necessary.
        :return: Nothing.
        """
        newpercent = int((self.iters / self.estimate) * 100)
        if newpercent > 100:
            msg = 'Unknown Progress'
            self.stdout.write('\r' + msg + ' '*(self.width-len(msg)))
        elif newpercent != self.percent or self.needswrite:
            prefix = _silent_format(self.prefix, newpercent)
            suffix = _silent_format(self.suffix, newpercent)
            navailable = self.width - len(prefix) - len(suffix)
            nblocks = int((self.iters / self.estimate) * navailable)
            self.stdout.write('\r' + prefix + self.block * nblocks + ' '*(
                navailable-nblocks) + suffix)
        self.needswrite = False

    def __next__(self):
        """
        Called on each iteration, to get a value.
        :return: The next value from self.it.
        """
        self.iters += 1
        self.__flush()
        self.__progress()
        self.percent = int((self.iters/self.estimate) * 100)
        try:
            return next(self.it)
        except StopIteration:
            if not self.finalized:
                # Write 100%
                self.iters = self.estimate
                self.needswrite = True
                self.__progress()

                # Replace stdout
                sys.stdout.close()
                sys.stdout = self.stdout
                print()

            # End the iteration
            raise StopIteration


def progress(it, *args, **kwargs):
    """
    Returns a progress bar if terminal is capable.

    See docstrings for Progress for more information on the other arguments.

    :param it: The iterator/list/range.
    :param args: Other positional arguments.
    :param kwargs: Other keyword arguments.
    :return:
    """
    termtype = get_term_type()
    if termtype == TermType.TTY or termtype == TermType.IPythonTerminal:
        return Progress(it, *args, **kwargs)
    else:
        return it


def progress_bar(size_param=(None, 'niters', 100)):
    """
    Turns a generator function into an iterator that uses Progress.

    Use this function as a decorator on a generator function.  Note that to use
    this function as a decorator, you must always call the function after the @
    sign.  That is, if you don't provide size_param, use
    :code:`@progress_bar()`, instead of :code:`@progress_bar`.

    :param size_param: This parameter tells the progress bar where it can
    find an estimate of the number of iterations in the parameter list of the
    call to the generator function.  The elements of the tuple are:
    - [0]: Argument index (default None)
    - [1]: Keyword argument index
    - [2]: Default (a better guess than util.Progress can provide).
    The default is to look for an niters parameter in the call to the wrapped
    generator.
    :return:
    """
    def wrap(f):
        """
        This function is returned by the call to progress_bar.

        When a function is decorated with progress_bar(), progress_bar()
        executes on module load.  It returns this function, which is the one
        that is actually called with the function to be decorated as a
        parameter.  This function returns a wrapped version of f.

        :param f: The actual function to be decorated.
        :return: The decorated/wrapped function.
        """
        def wrapped_f(*args, **kwargs):
            niters = size_param[2]
            if size_param[1] in kwargs:
                niters = kwargs[size_param[1]]
            elif size_param[0] is not None and size_param[0] < len(args):
                niters = args[size_param[0]]
            return progress(f(*args, **kwargs), niters=niters)
        return wrapped_f
    return wrap


def repeat_input(prompt='? ', in_type=str, num_retries=-1):
    """
    Get input of a specific type.  On failure, repeat.

    Prompt for a specific type of input.  If the input cannot be converted to
    that type, retry the input for the specified amount of retries.  If the
    retry amount is negative, the function will retry indefinitely.  If the
    parameter is omitted, it will be supplied as -1.

    If the user enters a blank string, this is considered a cancel sequence, and
    None is returned.  Similarly, if the number of retries is exceeded, None is
    returned.

    """
    rv_str = input(str(prompt))  # Ask for input the first time
    rv = None
    if rv_str == '':
        return None  # Blank strings mean return None
    try:
        rv = in_type(rv_str)  # Succeeds if the input was the right type
    except ValueError:
        while rv is None and num_retries != 0:
            try:
                # Continue retrying until we get a successful conversion
                num_retries -= 1
                rv_str = input('Bad Input.  ' + str(prompt))
                if rv_str == '':
                    return None
                rv = in_type(rv_str)
            except ValueError:
                pass
    return rv


class Menu:
    """
    A class that represents command line numeric menus.

    This Menu class offers a simple menu formulation that allows programmers to
    simply define an arbitrarily large or complex sequence of menus, simply by
    supplying, the title, options+actions, and any customization options they
    wish.  Actions may be instances of the Menu class, or they may be functions.
    Additionally, the Menu class allows you to create menus that re-appear after
    using an action.
    """

    def __init__(self, title='Main Menu', options=(), reentrant=False,
                 exit_text='Enter a blank string to exit.'):
        """Create an instance of the Menu class.

        title: (required) The title to be displayed
        options: A list of (string, action) tuples.
        reentrant: Return to the menu after executing an action?
        exit_text: The text to display for the exit option.

        """
        self.title = str(title)
        self.options = list(options)
        self.reentrant = reentrant
        self.exit_text = str(exit_text)

    # noinspection PyMethodMayBeStatic
    def pre_menu(self):
        """A pre-menu action for the menu."""
        print()

    # noinspection PyTypeChecker
    def display(self):
        """Run the menu."""
        running = True
        while running:
            running = self.reentrant  # only run once if not re-entrant

            # Display the menu title and options
            self.pre_menu()
            print(self.title)
            for number, option in enumerate(self.options, start=1):
                print('  %d. %s' % (number, option[0]))
            if self.exit_text is not None:
                print(self.exit_text)
            print('')

            selection = repeat_input('Selection: ', int)
            print()

            # If they enter a blank line, we assume they want to exit
            if selection is None:
                running = False
                continue

            # Check that the input is correct.  If not, run again.
            try:
                text, action = self.options[selection - 1]
            except IndexError:
                print('The option you selected is invalid.')
                running = True
                continue

            # Perform the action!
            if type(action) is Menu:
                action.display()
            elif type(action) is FunctionType:
                action()

    def add(self, name, action):
        """
        Add an item to the menu.

        :param name: Text to display for menu item.
        :param action: Action to perform (function or menu).
        """
        self.options.append((name, action))

    def add_function(self, name):
        """
        Function decorator to add directly to a menu.

        Put @menu.add_function('<text>') above a function to insert it right
        into the menu.

        :param name: The text to appear for the menu item.
        :return: A decorator that takes a function and adds it to the menu.
        """
        def decorator(f):
            self.add(name, f)
            return f
        return decorator
