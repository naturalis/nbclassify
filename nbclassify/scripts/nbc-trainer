#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script can be used to extract digital phenotypes from digital
photographs, export these to training data files, and train and test
artificial neural networks.

This script depends on configurations from a configuration file. See
config.yml for an example configuration file.

The script also depends on an SQLite database file with meta data for a
collection of digital photographs. This database is created by
harvest-images.py, which is also responsible for compiling the collection of
digital photographs.

The following subcommands are available:

* data: Create a tab separated file with training data.
* data-batch: Batch create tab separated files with training data.
* ann: Train an artificial neural network.
* ann-batch: Batch train artificial neural networks.
* test-ann: Test the performance of an artificial neural network.
* classify: Classify an image using an artificial neural network.

See the --help option for any of these subcommands for more information.
"""

import argparse
import logging
import os
import sys

import numpy as np

from nbclassify import conf, open_config
from nbclassify.db import session_scope
from nbclassify.exceptions import *

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data and train artificial neural "\
        "networks."
    )
    parser.add_argument(
        "conf",
        metavar="CONF_FILE",
        help="Path to a configurations file.")

    subparsers = parser.add_subparsers(
        help="Specify which task to start.",
        dest="task"
    )

    # Create an argument parser for sub-command 'meta'.
    help_meta = """Construct a meta data file for a directory of images.

    Images must be stored in a directory hierarchy, which is described in the
    configurations file. The meta data file is saved in the image directory.
    """

    parser_meta = subparsers.add_parser(
        "meta",
        help=help_meta,
        description=help_meta
    )
    parser_meta.add_argument(
        "imdir",
        metavar="PATH",
        help="Top most directory where images are stored in a directory " \
        "hierarchy.")

    # Create an argument parser for sub-command 'data'.
    help_data = """Create a tab separated file with training data.

    Preprocessing steps, features to extract, and a classification filter
    must be set in the configurations file.
    """

    parser_data = subparsers.add_parser(
        "data",
        help=help_data,
        description=help_data
    )
    parser_data.add_argument(
        "--cache-dir",
        metavar="PATH",
        required=True,
        help="Path to a directory where temporary data and cache is stored.")
    parser_data.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        required=True,
        help="Output file name for training data. Any existing file with " \
        "same name will be overwritten.")
    parser_data.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where Flickr harvested images are stored.")

    # Create an argument parser for sub-command 'data-batch'.
    help_data_batch = """Create training data for a classification hierarchy.

    Preprocessing steps, features to extract, and the classification
    hierarchy must be set in the configurations file.
    """

    parser_data_batch = subparsers.add_parser(
        "data-batch",
        help=help_data_batch,
        description=help_data_batch
    )
    parser_data_batch.add_argument(
        "--cache-dir",
        metavar="PATH",
        required=True,
        help="Path to a directory where temporary data and cache is stored.")
    parser_data_batch.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        required=True,
        help="Output directory where training data is stored.")
    parser_data_batch.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where Flickr harvested images are stored.")

    # Create an argument parser for sub-command 'ann'.
    help_ann = """Train an artificial neural network.

    Optional training parameters can be set in a configurations file.
    """

    parser_ann = subparsers.add_parser(
        "ann",
        help=help_ann,
        description=help_ann
    )
    parser_ann.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        required=True,
        help="Output file name for the artificial neural network. Any " \
        "existing file with the same name will be overwritten.")
    parser_ann.add_argument(
        "data",
        metavar="FILE",
        help="Path to tab separated file with training data.")

    # Create an argument parser for sub-command 'ann-batch'.
    help_ann_batch = """Train neural networks for a classification hierarchy.

    The classification hierarchy with optionally neural network training
    parameters for each level must be set in the configurations file.
    """

    parser_ann_batch = subparsers.add_parser(
        "ann-batch",
        help=help_ann_batch,
        description=help_ann_batch
    )
    parser_ann_batch.add_argument(
        "--data",
        "-d",
        metavar="PATH",
        required=True,
        help="Directory where the training data is stored.")
    parser_ann_batch.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        required=True,
        help="Output directory where the artificial neural networks are " \
        "stored.")
    parser_ann_batch.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where Flickr harvested images are stored.")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann = "Test an artificial neural network."

    parser_test_ann = subparsers.add_parser(
        "test-ann",
        help=help_test_ann,
        description=help_test_ann
    )
    parser_test_ann.add_argument(
        "--ann",
        "-a",
        metavar="FILE",
        required=True,
        help="A trained artificial neural network.")
    parser_test_ann.add_argument(
        "--error",
        "-e",
        metavar="N",
        type=float,
        default=0.00001,
        help="The maximum mean square error for classification. Default " \
        "is 0.00001")
    parser_test_ann.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output file name for the test results. Specifying this " \
        "option will output a table with the classification result for " \
        "each sample in TEST_DATA.")
    parser_test_ann.add_argument(
        "--test-data",
        "-t",
        metavar="FILE",
        required=True,
        help="Path to tab separated file containing test data.")
    parser_test_ann.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where Flickr harvested images are stored.")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann_batch = "Test the neural networks for a classification " \
    "hierarchy."

    parser_test_ann_batch = subparsers.add_parser(
        "test-ann-batch",
        help=help_test_ann_batch,
        description=help_test_ann_batch
    )
    parser_test_ann_batch.add_argument(
        "--anns",
        metavar="PATH",
        required=True,
        help="Directory where the artificial neural networks are stored.")
    parser_test_ann_batch.add_argument(
        "--error",
        "-e",
        metavar="N",
        type=float,
        default=0.00001,
        help="The maximum mean square error for classification. Default " \
        "is 0.00001")
    parser_test_ann_batch.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output the test results to FILE. Specifying this option " \
        "will output a table with the classification result for each " \
        "sample in the test data.")
    parser_test_ann_batch.add_argument(
        "--test-data",
        "-t",
        metavar="PATH",
        required=True,
        help="Directory where the test data is stored.")
    parser_test_ann_batch.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where Flickr harvested images are stored.")

    # Create an argument parser for sub-command 'classify'.
    help_classify = """Classify a digital photo.

    The classification filter must be set in the configurations file.
    """

    parser_classify = subparsers.add_parser(
        "classify",
        help=help_classify,
        description=help_classify
    )
    parser_classify.add_argument(
        "--ann",
        "-a",
        metavar="FILE",
        required=True,
        help="Path to a trained artificial neural network file.")
    parser_classify.add_argument(
        "--error",
        "-e",
        metavar="N",
        type=float,
        default=0.00001,
        help="The maximum error for classification. Default is 0.00001")
    parser_classify.add_argument(
        "--imdir",
        metavar="PATH",
        required=True,
        help="Base directory where Flickr harvested images are stored.")
    parser_classify.add_argument(
        "image",
        metavar="IMAGE_FILE",
        help="Path to image file to be classified.")

    # Create an argument parser for sub-command 'classify'.
    help_validate = """Test the performance of trained neural networks.

    Performs stratified K-fold cross validation.
    """

    parser_validate = subparsers.add_parser(
        "validate",
        help=help_validate,
        description=help_validate
    )
    parser_validate.add_argument(
        "--cache-dir",
        metavar="PATH",
        required=True,
        help="Path to a directory where temporary data and cache is stored.")
    parser_validate.add_argument(
        "--aivolver-config",
        metavar="PATH",
        help="Path to the configuration file for Aivolver. Using this option" \
        "results in training with Aivolver.")
    parser_validate.add_argument(
        "-k",
        metavar="N",
        type=int,
        default=3,
        help="The number of folds for the K-folds cross validation.")
    parser_validate.add_argument(
        "--autoskip",
        action='store_const',
        const=True,
        help="Skip the samples for which there are not at least `k` photos.")
    parser_validate.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where the Flickr harvested images are stored.")

    # Parse arguments.
    args = parser.parse_args()

    # Print debug messages if the -d flag is set for the Python interpreter.
    if sys.flags.debug:
        log_level = logging.DEBUG
        conf.debug = True
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format='%(levelname)s %(message)s')

    # Get path to meta data file.
    try:
        meta_path = os.path.join(args.imdir, conf.meta_file)
    except:
        meta_path = None

    # Load the configurations.
    config = open_config(args.conf)

    # Start selected task.
    try:
        if args.task == 'meta':
            meta(config, meta_path, args)
        if args.task == 'data':
            data(config, meta_path, args)
        if args.task == 'data-batch':
            data_batch(config, meta_path, args)
        elif args.task == 'ann':
            ann(config, args)
        elif args.task == 'ann-batch':
            ann_batch(config, meta_path, args)
        elif args.task == 'test-ann':
            test_ann(config, meta_path, args)
        elif args.task == 'test-ann-batch':
            test_ann_batch(config, meta_path, args)
        elif args.task == 'classify':
            classify(config, meta_path, args)
        elif args.task == 'validate':
            validate(config, meta_path, args)
    except ConfigurationError as e:
        logging.error("A configurational error was detected: %s", e)
        return 1
    except FileExistsError as e:
        logging.error("An output file already exists: %s", e)
        return 1
    except Exception as e:
        if conf.debug: raise
        logging.error(e)
        return 1

    return 0

def meta(config, meta_path, args):
    """Make meta data file for an image directory."""
    from nbclassify.db import MakeMeta, make_meta_db

    sys.stdout.write("Initializing database...\n")
    make_meta_db(meta_path)

    with session_scope(meta_path) as (session, metadata):
        mkmeta = MakeMeta(config, args.imdir)
        mkmeta.make(session, metadata)

def data(config, meta_path, args):
    """Start train data routines."""
    from nbclassify.data import PhenotypeCache, MakeTrainData

    try:
        filter_ = config.classification.filter.as_dict()
    except:
        raise ConfigurationError("The classification filter is not set")

    with session_scope(meta_path) as (session, metadata):
        cache = PhenotypeCache()
        cache.make(args.imdir, args.cache_dir, config, update=False)

        train_data = MakeTrainData(config, args.cache_dir)
        train_data.export(args.output, filter_)

def data_batch(config, meta_path, args):
    """Start batch train data routines."""
    from nbclassify.data import PhenotypeCache, BatchMakeTrainData

    with session_scope(meta_path) as (session, metadata):
        cache = PhenotypeCache()
        cache.make(args.imdir, args.cache_dir, config, update=False)

        train_data = BatchMakeTrainData(config, args.cache_dir)
        train_data.batch_export(args.output)

def ann(config, args):
    """Start neural network training routines."""
    from nbclassify.training import MakeAnn

    ann_maker = MakeAnn(config)
    ann_maker.train(args.data, args.output)

def ann_batch(config, meta_path, args):
    """Start batch neural network training routines."""
    from nbclassify.training import BatchMakeAnn

    with session_scope(meta_path) as (session, metadata):
        ann_maker = BatchMakeAnn(config)
        ann_maker.batch_train(args.data, args.output)

def test_ann(config, meta_path, args):
    """Start neural network testing routines."""
    from nbclassify.training import TestAnn

    with session_scope(meta_path) as (session, metadata):
        tester = TestAnn(config)
        tester.test(args.ann, args.test_data)

        if args.output:
            try:
                filter_ = config.classification.filter.as_dict()
            except:
                raise ConfigurationError("The classification filter is not set")

            tester.export_results(args.output, filter_, args.error)

def test_ann_batch(config, meta_path, args):
    """Start batch neural network testing routines."""
    from nbclassify.training import TestAnn

    with session_scope(meta_path) as (session, metadata):
        tester = TestAnn(config)
        tester.test_with_hierarchy(args.test_data, args.anns, args.error)

        if args.output:
            correct, total = tester.export_hierarchy_results(args.output)
            print "Correctly classified: {0}/{1} ({2:.2%})\n".\
                format(correct, total, float(correct)/total)

def classify(config, meta_path, args):
    """Start classification routines."""
    from nbclassify.classify import ImageClassifier
    from nbclassify.db import get_classes_from_filter
    from nbclassify.functions import get_classification, get_codewords

    try:
        filter_ = config.classification.filter.as_dict()
    except:
        raise ConfigurationError("The classification filter is not set")

    with session_scope(meta_path) as (session, metadata):
        classes = get_classes_from_filter(session, metadata, filter_)
        if not classes:
            raise ValueError("No classes found for filter `%s`" % filter_)
        codewords = get_codewords(classes)

        classifier = ImageClassifier(config)
        codeword = classifier.classify_image(args.image, args.ann, config)
        classification = get_classification(codewords, codeword, args.error)

    class_ = [class_ for mse,class_ in classification]
    print "Image is classified as {0}".format(", ".join(class_))

def validate(config, meta_path, args):
    """Start validation routines."""
    from nbclassify.data import PhenotypeCache
    from nbclassify.validate import Validator

    # Any existing training data or neural networks must be regenerated during
    # the validation process.
    conf.force_overwrite = True

    with session_scope(meta_path) as (session, metadata):
        cache = PhenotypeCache()
        cache.make(args.imdir, args.cache_dir, config, update=False)

        validator = Validator(config, args.cache_dir)
        if args.aivolver_config:
            validator.set_aivolver_config_path(args.aivolver_config)
        scores = validator.k_fold_xval_stratified(args.k, args.autoskip)

    print
    for path in sorted(scores.keys()):
        values = np.array(scores[path])

        print "Accuracy[{path}]: {mean:.2%} (+/- {sd:.2%})".format(**{
            'path': path,
            'mean': values.mean(),
            'sd': values.std() * 2
        })

if __name__ == "__main__":
    main()
