#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classify digital photos using artificial neural networks.

The photo is classified on different levels in a classification hierarchy,
for example a taxanomic hierarchy. A different neural network may be used to
classify at each level in the hierarchy. The classification hierarchy is
set in a configuration file. See trainer.yml for an example configuration
file with the classification hierarchy set.

The neural networks on which this script depends are created by a separate
script, trainer.py. See `trainer.py batch-data --help` and
`trainer.py batch-ann --help` for more information.

The script also depends on an SQLite database file with meta data for a
collection of digital photographs. This database is created by
harvest-images.py, which is also responsible for compiling the collection of
digital photographs.

See the --help option for more information.
"""

import argparse
from contextlib import contextmanager
import hashlib
import logging
import os
import sys

import cv2
import imgpheno as ft
import numpy as np
from pyfann import libfann
import sqlalchemy
import yaml

import nbclassify as nbc

GREEN = '\033[32m'
GREEN_BOLD = '\033[1;32m'
RED = '\033[31m'
RED_BOLD = '\033[1;31m'

def main():
    # Setup the argument parser.
    parser = argparse.ArgumentParser(description="Classify digital " \
        "photographs using a committee of artificial neural networks.")

    parser.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with the classification " \
        "hierarchy set.")
    parser.add_argument("--db", metavar="DB", required=True,
        help="Path to a database file with photo meta data.")
    parser.add_argument("--anns", metavar="PATH", required=True,
        help="Path to a directory containing the neural networks.")
    parser.add_argument("--error", metavar="N", type=float, default=0.0001,
        help="The maximum error for classification at each level. Default " \
        "is 0.0001. If the maximum error for a level is set in the " \
        "classification hierarchy, then that value is used instead.")
    parser.add_argument("--verbose", "-v", action='store_const',
        const=True, help="Explain what is being done.")
    parser.add_argument("--color", action='store_const',
        const=True, help="Show colored results. Only works on terminals " \
        "that support ANSI escape sequences.")
    parser.add_argument("images", metavar="PATH", nargs='+',
        help="Path to a digital photograph to be classified.")

    # Parse arguments.
    args = parser.parse_args()

    # Print debug messages if the -d flag is set for the Python interpreter.
    if sys.flags.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level, format='%(levelname)s %(message)s')

    config = open_yaml(args.conf)
    classifier = ImageClassifier(config, args.db)
    classifier.set_error(args.error)

    for image_path in args.images:
        classify_image(classifier, image_path, args.anns, args.color)

def classify_image(classifier, image_path, anns_dir, use_color=False):
    print "Image: %s" % image_path

    try:
        classes, errors = classifier.classify_with_hierarchy(image_path, anns_dir)
    except Exception as e:
        logging.error(e)
        return

    # Check for failed classification.
    if not classes[0]:
        print "  Classification:"
        print "    %s" % ansi_colored("Failed", RED_BOLD, not use_color)
        return

    # Calculate the mean square error for each classification path.
    errors_classes = [(sum(e)/len(e),c) for e,c in zip(errors, classes)]

    # Get the level names.
    levels = classifier.get_classification_hierarchy_levels()

    # Print the classification results, sorted by error.
    for error, classes_ in sorted(errors_classes):
        print "  Classification:"
        for i, (level, class_) in enumerate(zip(levels, classes_)):
            # Make class an empty string if it is None.
            class_ = class_ if class_ is not None else ''

            print "    %s%s: %s" % (
                '  ' * i,
                level,
                ansi_colored(class_, GREEN_BOLD, not use_color)
            )
        print "    Mean square error: %s" % error

def ansi_colored(s, color, raw=False):
    """Return an ANSI colored version of string `s`.

    The string is formatted with ANSI escape character `color`. Returns the
    raw string if `raw` is True or if the string evaluates to False.
    """
    if raw or not s:
        return s
    replace = {
        'color': color,
        'reset': '\033[0m',
        's': s
    }
    return "{color}{s}{reset}".format(**replace)

@contextmanager
def session_scope(db_path):
    """Provide a transactional scope around a series of operations."""
    engine = sqlalchemy.create_engine('sqlite:///%s' % os.path.abspath(db_path),
        echo=sys.flags.debug)
    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    session = Session()
    metadata = sqlalchemy.MetaData()
    metadata.reflect(bind=engine)
    try:
        yield (session, metadata)
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def get_object(d):
    """Return an object from a dictionary."""
    if not isinstance(d, dict):
        raise TypeError("Argument 'd' is not a dictionary")
    return nbc.Struct(d)

def open_yaml(path):
    """Read a YAML file and return as an object."""
    with open(path, 'r') as f:
        config = yaml.load(f)
    return get_object(config)

class ImageClassifier(nbc.Common):
    """Classify an image."""

    def __init__(self, config, db_path):
        super(ImageClassifier, self).__init__(config)

        self.set_db_path(db_path)
        self.error = 0.0001
        self.cache = {}

        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("missing `classification.hierarchy`")

        # Get the classification hierarchy from the database.
        with session_scope(db_path) as (session, metadata):
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

    def set_db_path(self, path):
        """Set the path to the database file."""
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        self.db_path = path

    def set_error(self, error):
        """Set the default maximum error for classification."""
        if not 0 < error < 1:
            raise ValueError("Error must be a value between 0 and 1" % error)
        self.error = error

    def get_classification_hierarchy_levels(self):
        """Return the list of level names from the classification hierarchy."""
        return [l.name for l in self.class_hr]

    def classify_image(self, im_path, ann_path, config):
        """Classify an image file and return the codeword.

        Preprocess and extract features from the image `im_path` as defined
        in the configuration object `config`, and use the features as input
        for the neural network `ann_path` to obtain a codeword.
        """
        if not os.path.isfile(im_path):
            raise IOError("Cannot open %s (no such file)" % im_path)
        if not os.path.isfile(ann_path):
            raise IOError("Cannot open %s (no such file)" % ann_path)
        if 'preprocess' not in config:
            raise nbc.ConfigurationError("missing `preprocess`")
        if 'features' not in config:
            raise nbc.ConfigurationError("missing `features`")

        ann = libfann.neural_net()
        ann.create_from_file(str(ann_path))

        # Get the MD5 hash.
        hasher = hashlib.md5()
        with open(im_path, 'rb') as fh:
            buf = fh.read()
            hasher.update(buf)

        # Create a hash of the feature extraction options.
        hash_ = "%s.%s.%s" % (hasher.hexdigest(), hash(config.preprocess), hash(config.features))

        if hash_ in self.cache:
            phenotype = self.cache[hash_]
        else:
            phenotyper = nbc.Phenotyper()
            phenotyper.set_image(im_path)
            phenotyper.set_config(config)
            phenotype = phenotyper.make()

            # Cache the phenotypes, in case they are needed again.
            self.cache[hash_] = phenotype

        logging.info("Using ANN `%s`" % ann_path)
        codeword = ann.run(phenotype)

        return codeword

    def classify_with_hierarchy(self, image_path, ann_base_path=".",
                                path=[], path_error=[]):
        """Start recursive classification.

        Classify the image `image_path` with neural networks from the
        directory `ann_base_path`. The image is classified for each level
        in the classification hierarchy ``classification.hierarchy`` set in
        the configurations file. Each level can use a different neural
        network for classification; the file names for the neural networks
        are set in ``classification.hierarchy[n].ann_file``. Multiple
        classifications are returned if the classification of a level in
        the hierarchy returns multiple classifications, in which case the
        classification path is split into multiple classifications paths.

        Returns a pair of tuples ``(classifications, errors)``, the list
        of classifications, and the list of errors for each classification.
        Each classification is a list of the classes for each level in the
        hierarchy, top to bottom. The list of errors has the same dimension
        of the list of classifications, where each value corresponds to the
        mean square error of each classification.
        """
        levels = self.get_classification_hierarchy_levels()
        paths = []
        paths_errors = []

        if len(path) == len(levels):
            return ([path], [path_error])
        elif len(path) > len(levels):
            raise ValueError("Classification hierarchy depth exceeded")

        # Get the level specific configurations.
        level = conf = self.class_hr[len(path)]

        # Replace any placeholders in the ANN path.
        ann_file = level.ann_file
        for key, val in zip(levels, path):
            val = val if val is not None else '_'
            ann_file = ann_file.replace("__%s__" % key, val)

        # Get the class names for this node in the taxonomic hierarchy.
        level_classes = self.get_childs_from_hierarchy(self.taxon_hr, path)

        # Some levels must have classes set.
        if level_classes == [None] and level.name in ('genus','species'):
            raise ValueError("Classes for level `%s` are not set" % level.name)

        if level_classes == [None]:
            # No need to classify if there are no classes set.
            classes = level_classes
            class_errors = (0.0,)
        else:
            # Get the codewords for the classes.
            class_codewords = self.get_codewords(level_classes)

            # Classify the image and obtain the codeword.
            ann_path = os.path.join(ann_base_path, ann_file)
            codeword = self.classify_image(image_path, ann_path, conf)

            # Set the maximum classification error for this level.
            try:
                max_error = level.max_error
            except:
                max_error = self.error

            # Get the class name associated with this codeword.
            classes = self.get_classification(class_codewords,
                codeword, max_error)
            if classes:
                class_errors, classes = zip(*classes)
            else:
                class_errors = classes = []

            # Test branching.
            #if level.name == 'section':
                #classes += ('Coryopedilum','Brachypetalum')
                #class_errors += (0.0001,0.0002)
            #if level.name == 'species' and path[-1] == 'Paphiopedilum':
                #classes += ('druryi',)
                #class_errors += (0.0003,)
            #if level.name == 'species' and path[-1] == 'Coryopedilum':
                #classes = ()
                #class_errors = ()

        # Print some info messages.
        path_s = [str(p) for p in path]
        path_s = '/'.join(path_s)

        if len(classes) == 0:
            logging.info("Failed to classify on level `%s` at node `/%s`" % (
                level.name,
                path_s)
            )
            return ([path], [path_error])
        elif len(classes) > 1:
            logging.info("Branching in level `%s` at node '/%s' into `%s`" % (
                level.name,
                path_s,
                ', '.join(classes))
            )
        else:
            logging.info("Level `%s` at node `/%s` classified as `%s`" % (
                level.name,
                path_s,
                classes[0])
            )

        for class_, mse in zip(classes, class_errors):
            # Recurse into lower hierarchy levels.
            paths_, paths_errors_ = self.classify_with_hierarchy(image_path,
                ann_base_path, path+[class_], path_error+[mse])

            # Keep a list of each classification path and their
            # corresponding errors.
            paths.extend(paths_)
            paths_errors.extend(paths_errors_)

        assert len(paths) == len(paths_errors), \
            "Number of paths must be equal to the number of path errors"

        return paths, paths_errors

if __name__ == "__main__":
    main()
