# -*- coding: utf-8 -*-

"""General functions."""

from argparse import Namespace
import os
import shutil
import tempfile

import yaml

def classification_hierarchy_filters(levels, hr, path=[]):
    """Return the classification filter for each path in a hierarchy.

    Returns the classification filter for each possible path in the
    hierarchy `hr`. The name of each level in the hierarchy must be set in
    the list `levels`. The list `path` holds the position in the
    hierarchy. An empty `path` means start at the root of the hierarchy.
    Filters that return no classes are not returned.

    Example:

        >>> hr = {
        ...     'Selenipedium': {
        ...         None: ['palmifolium']
        ...     },
        ...     'Phragmipedium': {
        ...         'Micropetalum': ['andreettae', 'besseae'],
        ...         'Lorifolia': ['boissierianum', 'brasiliense']
        ...     }
        ... }
        >>> levels = ['genus', 'section', 'species']
        >>> filters = cmn.classification_hierarchy_filters(levels, hr)
        >>> for f in filters:
        ...     print f
        ...
        {'where': {}, 'class': 'genus'}
        {'where': {'section': None, 'genus': 'Selenipedium'}, 'class': 'species'}
        {'where': {'genus': 'Phragmipedium'}, 'class': 'section'}
        {'where': {'section': 'Lorifolia', 'genus': 'Phragmipedium'}, 'class': 'species'}
        {'where': {'section': 'Micropetalum', 'genus': 'Phragmipedium'}, 'class': 'species'}

    These filters are used directly by methods like
    :meth:`db.get_filtered_photos_with_taxon` and
    :meth:`db.get_classes_from_filter`.
    """
    filter_ = {}

    # The level number that is being classfied (0 based).
    level_no = len(path)

    if level_no > len(levels) - 1:
        raise ValueError("Maximum classification hierarchy depth exceeded")

    # Set the level to classify on.
    filter_['class'] = levels[level_no]

    # Set the where fields.
    filter_['where'] = {}
    for i, class_ in enumerate(path):
        name = levels[i]
        filter_['where'][name] = class_

    # Get the classes for the current hierarchy path.
    classes = get_childs_from_hierarchy(hr, path)

    # Only return the filter if the classes are set.
    if classes != [None]:
        yield filter_

    # Stop iteration if the last level was classified.
    if level_no == len(levels) - 1:
        return

    # Recurse into lower hierarchy levels.
    for c in classes:
        for f in classification_hierarchy_filters(levels, hr,
                path+[c]):
            yield f

def delete_temp_dir(path, recursive=False):
    """Delete a temporary directory.

    As a safeguard, this function only removes directories and files that are
    within the system's temporary directory (e.g. /tmp on Unix). Setting
    `recursive` to True also deletes its contents.
    """
    path = os.path.abspath(str(path))
    tmp_dir = tempfile.gettempdir()
    if os.path.isdir(path):
        if path.startswith(tmp_dir):
            if recursive:
                shutil.rmtree(path)
            else:
                os.rmdir(path)
        else:
            raise ValueError("Path is not a subdirectory of %s" % tmp_dir)

def get_childs_from_hierarchy(hr, path=[]):
    """Return the child node names for a node in a hierarchy.

    Returns a list of child node names of the hierarchy `hr` at node with
    the path `path`. The hierarchy `hr` is a nested dictionary, as
    returned by :meth:`db.get_taxon_hierarchy`. Which node to get the childs
    from is specified by `path`, which is a list of the node names up to
    that node. An empty list for `path` means the names of the nodes of
    the top level are returned.

    Example:

        >>> hr = {
        ...     'Phragmipedium': {
        ...         'Micropetalum': ['andreettae', 'besseae'],
        ...         'Lorifolia': ['boissierianum', 'brasiliense']
        ...     },
        ...     'Selenipedium': {
        ...         None: ['palmifolium']
        ...     }
        ... }
        >>> cmn.get_childs_from_hierarchy(hr)
        ['Selenipedium', 'Phragmipedium']
        >>> cmn.get_childs_from_hierarchy(hr, ['Phragmipedium'])
        ['Lorifolia', 'Micropetalum']
        >>> cmn.get_childs_from_hierarchy(hr, ['Phragmipedium','Micropetalum'])
        ['andreettae', 'besseae']
    """
    nodes = hr.copy()
    try:
        for name in path:
            nodes = nodes[name]
    except:
        raise ValueError("No such path `%s` in the hierarchy" % \
            '/'.join(path))

    if isinstance(nodes, dict):
        names = nodes.keys()
    elif isinstance(nodes, list):
        names = nodes
    else:
        raise ValueError("Incorrect hierarchy format")

    return names

def get_classification(codewords, codeword, error=0.01, on=1.0):
    """Return the human-readable classification for a codeword.

    Each bit in the codeword `codeword` is compared to the `on` bit in
    each of the codewords in `codewords`, which is a dictionary of the
    format ``{class: codeword, ..}``. If the mean square error for a bit
    is less than or equal to `error`, then the corresponding class is
    assigned to the codeword. So it is possible that a codeword is
    assigned to multiple classes.

    The result is returned as an error-sorted list of 2-tuples ``[(error,
    class), ..]``. Returns an empty list if no classes were found.
    """
    if len(codewords) != len(codeword):
        raise ValueError("Codeword size mismatch. The number of " \
            "codeword bits does not match the number of classes. " \
            "The classes in the meta data file must match the classes " \
            "used to train the neural networks.")
    classes = []
    for class_, word in codewords.items():
        for i, bit in enumerate(word):
            if bit == on:
                mse = (float(bit) - codeword[i]) ** 2
                if mse <= error:
                    classes.append((mse, class_))
                break
    return sorted(classes)

def get_codewords(classes, on=1, off=-1):
    """Return codewords for a list of classes.

    Takes a list of class names `classes`. The class list is sorted, and a
    codeword is created for each class. Each codeword is a list of
    ``len(classes)`` bits, where all bits have an `off` value, except for
    one, which has an `on` value. Returns a dictionary ``{class: codeword,
    ..}``. The same set of classes always returns the same codeword for
    each class.
    """
    n = len(classes)
    codewords = {}
    for i, class_ in enumerate(sorted(classes)):
        cw = [off] * n
        cw[i] = on
        codewords[class_] = cw
    return codewords

def open_config(path):
    """Read a configurations file and return as a nested :class:`Struct` object.

    The configurations file is in the YAML format and is loaded from file path
    `path`.
    """
    with open(path, 'r') as f:
        config = yaml.load(f)
    return Struct(config)

def path_from_filter(filter_, levels):
    """Return the path from a classification filter.

    Example:

        >>> f = {'where': {'section': 'Micropetalum', 'genus': 'Phragmipedium'},
        ... 'class': 'species'}
        >>> path_from_filter(f, ('genus','section','species'))
        ['Phragmipedium', 'Micropetalum']
    """
    path = []
    for name in levels:
        try:
            path.append(filter_['where'][name])
        except:
            return path
    return path

def filter_is_valid(f):
    """Return True if the classification filter `f` is valid."""
    if not isinstance(f, dict):
        return False
    if 'class' not in f:
        return False
    for key in f:
        if key not in ('where', 'class'):
            return False
    return True


class Struct(Namespace):

    """Return a dictionary as an object."""

    def __init__(self, d):
        if not isinstance(d, dict):
            raise TypeError("Expected a dictionary, got {0} instead".\
                format(type(d)))
        for key, val in d.iteritems():
            if isinstance(val, (list, tuple)):
                setattr(self, str(key), [self.__class__(x) if \
                    isinstance(x, dict) else x for x in val])
            elif isinstance(val, self.__class__):
                setattr(self, str(key), val)
            else:
                setattr(self, str(key), self.__class__(val) if \
                    isinstance(val, dict) else val)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __getitem__(self, key):
        return getattr(self, str(key))

    def as_dict(self):
        d = vars(self)
        for key, val in d.iteritems():
            if isinstance(val, (list, tuple)):
                d[key] = [x.as_dict() if \
                    isinstance(x, self.__class__) else x for x in val]
            elif isinstance(val, self.__class__):
                d[key] = val.as_dict()
            else:
                d[key] = val
        return d
