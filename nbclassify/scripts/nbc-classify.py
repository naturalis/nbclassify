#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classify digital photos using artificial neural networks.

The photo is classified on different levels in a classification hierarchy,
for example a taxanomic hierarchy. A different neural network may be used to
classify at each level in the hierarchy. The classification hierarchy is
set in a configuration file. See config.yml for an example configuration
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

from nbclassify import conf, open_config
from nbclassify.classify import ImageClassifier

# File name of the meta data file.
META_FILE = conf.meta_file

# ANSI colors.
GREEN = '\033[32m'
GREEN_BOLD = '\033[1;32m'
RED = '\033[31m'
RED_BOLD = '\033[1;31m'

def main():
    parser = argparse.ArgumentParser(
        description="Classify digital photographs using a committee of " \
        "artificial neural networks."
    )
    parser.add_argument(
        "--conf",
        metavar="FILE",
        required=True,
        help="Path to a configurations file with the classification " \
        "hierarchy.")
    parser.add_argument(
        "--imdir",
        metavar="PATH",
        required=True,
        help="Base directory where Flickr harvested images are stored.")
    parser.add_argument(
        "--anns",
        metavar="PATH",
        required=True,
        help="Path to a directory containing the neural networks for " \
        "a classification hierarchy.")
    parser.add_argument(
        "--error",
        metavar="N",
        type=float,
        default=0.0001,
        help="The maximum error for classification at each level. Default " \
        "is 0.0001. If the maximum error for a level is set in the " \
        "classification hierarchy, then that value is used instead.")
    parser.add_argument(
        "--verbose",
        "-v",
        action='store_const',
        const=True,
        help="Explain what is being done.")
    parser.add_argument(
        "--color",
        action='store_const',
        const=True,
        help="Show colored results. Only works on terminals " \
        "that support ANSI escape sequences.")
    parser.add_argument("images",
        metavar="PATH",
        nargs='+',
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

    # Get path to meta data file.
    meta_path = os.path.join(args.imdir, META_FILE)

    config = open_config(args.conf)
    classifier = ImageClassifier(config, meta_path)
    classifier.set_error(args.error)

    for image_path in args.images:
        classify_image(classifier, image_path, args.anns, args.color)

def classify_image(classifier, image_path, anns_dir, use_color=False):
    print "Image: %s" % image_path

    classes, errors = classifier.classify_with_hierarchy(image_path, anns_dir)

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

if __name__ == "__main__":
    main()
