#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script can be used to extract digital phenotypes from digital
photographs, export these to training data files, and train and test
artificial neural networks.

This script depends on configurations from a configuration file. See
trainer.yml for an example configuration file.

The script also depends on an SQLite database file with meta data for a
collection of digital photographs. This database is created by
harvest-images.py, which is also responsible for compiling the collection of
digital photographs.

The following subcommands are available:

* data: Create a tab separated file with training data.
* batch-data: Batch create tab separated files with training data.
* ann: Train an artificial neural network.
* batch-ann: Batch train artificial neural networks.
* test-ann: Test the performance of an artificial neural network.
* classify: Classify an image using an artificial neural network.

See the --help option for any of these subcommands for more information.
"""

import argparse
from contextlib import contextmanager
import logging
import os
import re
import sys

import cv2
import numpy as np
from pyfann import libfann
import sqlalchemy
import yaml

import nbclassify as nbc
# Import the feature extraction library.
# https://github.com/naturalis/feature-extraction
import features as ft

OUTPUT_PREFIX = "OUT:"

def main():
    # Print debug messages if the -d flag is set for the Python interpreter.
    # Otherwise just show log messages of type INFO.
    if sys.flags.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Setup the argument parser.
    parser = argparse.ArgumentParser(description="Generate training data " \
        "and train artificial neural networks.")
    subparsers = parser.add_subparsers(help="Specify which task to start.",
        dest='task')

    # Create an argument parser for sub-command 'data'.
    help_data = """Create a tab separated file with training data.
    Preprocessing steps, features to extract, and a classification filter
    must be set in a configurations file. See trainer.yml for an example."""

    parser_data = subparsers.add_parser('data',
        help=help_data, description=help_data)
    parser_data.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with feature extraction parameters.")
    parser_data.add_argument("--db", metavar="DB",
        help="Path to a database file with photo meta data. If omitted " \
        "this defaults to a file photos.db in the photo's directory.")
    parser_data.add_argument("--output", "-o", metavar="FILE", required=True,
        help="Output file name for training data. Any existing file with " \
        "same name will be overwritten.")
    parser_data.add_argument("basepath", metavar="PATH",
        help="Base directory where to look for photo's. The database file" \
        "with photo meta data will be used to find photo's in this directory.")

    # Create an argument parser for sub-command 'batch-data'.
    help_batch_data = """Batch create tab separated files with training
    data. Preprocessing steps, features to extract, and the classification
    hierarchy must be set in a configurations file, See trainer.yml for an
    example."""

    parser_batch_data = subparsers.add_parser('batch-data',
        help=help_batch_data, description=help_batch_data)
    parser_batch_data.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with feature extraction parameters.")
    parser_batch_data.add_argument("--db", metavar="DB",
        help="Path to a database file with photo meta data. If omitted " \
        "this defaults to a file photos.db in the photo's directory.")
    parser_batch_data.add_argument("--output", "-o", metavar="PATH",
        required=True,
        help="Output directory where training data is stored.")
    parser_batch_data.add_argument("basepath", metavar="PATH",
        help="Base directory where to look for photo's. The database file" \
        "with photo meta data will be used to find photo's in this directory.")

    # Create an argument parser for sub-command 'ann'.
    help_ann = """Train an artificial neural network. Optional training
    parameters `ann` can be set in a separate configurations file. See
    trainer.yml for an example."""

    parser_ann = subparsers.add_parser('ann',
        help=help_ann, description=help_ann)
    parser_ann.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with ANN training parameters.")
    parser_ann.add_argument("--epochs", metavar="N", type=float,
        help="Maximum number of epochs. Overwrites value in --conf.")
    parser_ann.add_argument("--error", metavar="N", type=float,
        help="Desired mean square error on training data. Overwrites value " \
        "in --conf.")
    parser_ann.add_argument("--output", "-o", metavar="FILE", required=True,
        help="Output file name for the artificial neural network. Any " \
        "existing file with same name will be overwritten.")
    parser_ann.add_argument("data", metavar="TRAIN_DATA",
        help="Path to tab separated file with training data.")

    # Create an argument parser for sub-command 'batch-ann'.
    help_batch_ann = """Batch train a committee of artificial neural
    networks. The classification hierarchy with optionally neural network
    training parameters for each level must be set in a configurations
    file. See trainer.yml for an example."""

    parser_batch_ann = subparsers.add_parser('batch-ann',
        help=help_batch_ann, description=help_batch_ann)
    parser_batch_ann.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with ANN training parameters.")
    parser_batch_ann.add_argument("--db", metavar="DB", required=True,
        help="Path to a database file with photo meta data.")
    parser_batch_ann.add_argument("--data", metavar="PATH", required=True,
        help="Directory where the training data is stored.")
    parser_batch_ann.add_argument("--epochs", metavar="N", type=float,
        help="Maximum number of epochs. Overwrites value in --conf.")
    parser_batch_ann.add_argument("--error", metavar="N", type=float,
        help="Desired mean square error on training data. Overwrites value " \
        "in --conf.")
    parser_batch_ann.add_argument("--output", "-o", metavar="PATH", required=True,
        help="Output directory where the artificial neural networks are stored.")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann = """Test an artificial neural network. If `--output` is
    used, then --db must be set, and the classification filter must be set
    in the configurations file. See trainer.yml for an example."""

    parser_test_ann = subparsers.add_parser('test-ann',
        help=help_test_ann, description=help_test_ann)
    parser_test_ann.add_argument("--ann", metavar="FILE", required=True,
        help="A trained artificial neural network.")
    parser_test_ann.add_argument("--db", metavar="DB",
        help="Path to a database file with photo meta data. Must be " \
        "used together with --output.")
    parser_test_ann.add_argument("--output", "-o", metavar="FILE",
        help="Output file name for the test results. Specifying this " \
        "option will output a table with the classification result for " \
        "each sample in TEST_DATA.")
    parser_test_ann.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with a classification filter.")
    parser_test_ann.add_argument("--error", metavar="N", type=float,
        default=0.00001,
        help="The maximum mean square error for classification. Default " \
        "is 0.00001")
    parser_test_ann.add_argument("data", metavar="TEST_DATA",
        help="Path to tab separated file containing test data.")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann_batch = """Test the artificial neural networks for a classification hierarchy. See trainer.yml for an example."""

    parser_test_ann_batch = subparsers.add_parser('test-ann-batch',
        help=help_test_ann_batch, description=help_test_ann_batch)
    parser_test_ann_batch.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with a classification filter.")
    parser_test_ann_batch.add_argument("--db", metavar="DB",
        help="Path to a database file with photo meta data. Must be " \
        "used together with --output.")
    parser_test_ann_batch.add_argument("--test-data", metavar="PATH",
        required=True,
        help="Directory where the test data is stored.")
    parser_test_ann_batch.add_argument("--anns", metavar="PATH",
        required=True,
        help="Directory where the artificial neural networks are stored.")
    parser_test_ann_batch.add_argument("--error", metavar="N", type=float,
        default=0.00001,
        help="The maximum mean square error for classification. Default " \
        "is 0.00001")
    parser_test_ann_batch.add_argument("--output", "-o", metavar="FILE",
        help="Output the test results to FILE. Specifying this " \
        "option will output a table with the classification result for " \
        "each sample in the test data.")

    # Create an argument parser for sub-command 'classify'.
    help_classify = """Classify a digital photo. The classification filter
    must be set in the configurations file. See trainer.yml for an example."""

    parser_classify = subparsers.add_parser('classify',
        help=help_classify, description=help_classify)
    parser_classify.add_argument("--ann", metavar="FILE", required=True,
        help="Path to a trained artificial neural network file.")
    parser_classify.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with a classification filter set.")
    parser_classify.add_argument("--db", metavar="DB", required=True,
        help="Path to a database file with photo meta data.")
    parser_classify.add_argument("--error", metavar="N", type=float,
        default=0.00001,
        help="The maximum error for classification. Default is 0.00001")
    parser_classify.add_argument("image", metavar="IMAGE",
        help="Path to image file to be classified.")

    # Parse arguments.
    args = parser.parse_args()

    if args.task == 'data':
        # Set the default database path if not set.
        if args.db is None:
            args.db = os.path.join(args.basepath, 'photos.db')

        config = open_yaml(args.conf)

        try:
            filter_ = config.classification.filter
        except:
            logging.error("The configuration file is missing `classification.filter`")
            return

        try:
            train_data = MakeTrainData(config, args.basepath, args.db)
            train_data.export(args.output, filter_, config)
        except Exception as e:
            logging.error(e)
            raise

    if args.task == 'batch-data':
        # Set the default database path if not set.
        if args.db is None:
            args.db = os.path.join(args.basepath, 'photos.db')

        try:
            config = open_yaml(args.conf)
            train_data = BatchMakeTrainData(config, args.basepath, args.db)
            train_data.batch_export(args.output)
        except Exception as e:
            logging.error(e)
            raise

    elif args.task == 'ann':
        try:
            config = open_yaml(args.conf)
            ann_maker = MakeAnn(config, args, config.ann)
            ann_maker.train(args.data, args.output)
        except Exception as e:
            logging.error(e)

    elif args.task == 'batch-ann':
        try:
            config = open_yaml(args.conf)
            ann_maker = BatchMakeAnn(config, args.db, args)
            ann_maker.batch_train(args.data, args.output)
        except Exception as e:
            logging.error(e)
            raise

    elif args.task == 'test-ann':
        config = open_yaml(args.conf)
        tester = TestAnn(config)
        tester.test(args.ann, args.data)
        if args.output:
            if not args.db:
                sys.exit("Option --output must be used together with --db")

            try:
                filter_ = config.classification.filter
            except:
                raise nbc.ConfigurationError("missing `classification.filter`")

            tester.export_results(args.output, args.db, filter_, args.error)

    elif args.task == 'test-ann-batch':
        config = open_yaml(args.conf)
        tester = TestAnn(config)
        tester.test_with_hierarchy(args.db, args.test_data, args.anns, args.error)

        if args.output:
            total, correct = tester.export_hierarchy_results(args.output)
            sys.stderr.write("Correctly classified: {0}/{1} ({2:.2%})\n".format(correct, total, float(correct)/total))

    elif args.task == 'classify':
        config = open_yaml(args.conf)
        classifier = ImageClassifier(config, args.ann, args.db)
        class_ = classifier.classify(args.image, args.error)
        class_ = [class_ for mse,class_ in class_ann]
        logging.info("Image is classified as %s" % ", ".join(class_))

    sys.exit()

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

class MakeTrainData(nbc.Common):
    """Generate training data."""

    def __init__(self, config, base_path, db_path):
        """Constructor for training data generator.

        Expects a configurations object `config`, a path to the root
        directory of the photos, and a path to the database file `db_path`
        containing photo meta data.
        """
        super(MakeTrainData, self).__init__(config)
        self.set_base_path(base_path)
        self.set_db_path(db_path)

    def set_base_path(self, path):
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.base_path = path

    def set_db_path(self, path):
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        self.db_path = path

    def export(self, filename, filter_, config, overwrite=False):
        """Write the training data to `filename`.

        Images to be processed are obtained from the database. Which images
        are obtained and with which classes is set by the filter
        `filter_`. A configuration object `config` denotes which
        preprocessing steps to take and what features are to be extracted.
        """
        if not overwrite and os.path.isfile(filename):
            raise nbc.FileExistsError("Output file %s already exists." % filename)

        # Get list of image paths and corresponding classifications from
        # the meta data database.
        with session_scope(self.db_path) as (session, metadata):
            images = self.get_photos_with_class(session, metadata, filter_)

        if len(images) == 0:
            logging.info("No images found for the filter `%s`" % filter_)
            return

        logging.info("Going to process %d photos..." % len(images))

        # Get a unique list of classes.
        classes = set([x[1] for x in images])

        # Make codeword for each class.
        codewords = self.get_codewords(classes)

        # Construct the header.
        header_data, header_out = self.__make_header(len(classes))
        header = ["ID"] + header_data + header_out

        # Generate the training data.
        with open(filename, 'w') as fh:
            # Write the header.
            fh.write( "%s\n" % "\t".join(header) )

            # Set the training data.
            training_data = nbc.TrainData(len(header_data), len(classes))
            phenotyper = nbc.Phenotyper()
            failed = []
            for im_path, im_class in images:
                logging.info("Processing `%s` of class `%s`..." % (im_path, im_class))
                im_path_real = os.path.join(self.base_path, im_path)

                try:
                    phenotyper.set_image(im_path_real)
                    phenotyper.set_config(config)
                except:
                    logging.warning("Failed to read `%s`. Skipping." % im_path_real)
                    failed.append(im_path_real)
                    continue

                # Create a phenotype from the image.
                phenotype = phenotyper.make()

                assert len(phenotype) == len(header_data), "Data length mismatch"

                training_data.append(phenotype, codewords[im_class], label=im_path)

            training_data.finalize()

            # Round all values.
            training_data.round_input(6)

            # Write data rows.
            for label, input, output in training_data:
                row = []
                row.append(label)
                row.extend(input.astype(str))
                row.extend(output.astype(str))
                fh.write("%s\n" % "\t".join(row))

        logging.info("Training data written to %s" % filename)

        # Print list of failed objects.
        if len(failed) > 0:
            logging.warning("Some files could not be processed:")
            for path in failed:
                logging.warning("- %s" % path)

    def __make_header(self, n_out):
        """Construct a header from features configurations.

        Header is returned as a 2-tuple ``(data_columns, output_columns)``.
        """
        if 'features' not in self.config:
            raise nbc.ConfigurationError("missing `features`")

        data = []
        out = []
        for feature, args in vars(self.config.features).iteritems():
            if feature == 'color_histograms':
                for colorspace, bins in vars(args).iteritems():
                    for ch, n in enumerate(bins):
                        for i in range(1, n+1):
                            data.append("%s:%d" % (colorspace[ch], i))

            if feature == 'color_bgr_means':
                bins = getattr(args, 'bins', 20)
                for i in range(1, bins+1):
                    for axis in ("HOR", "VER"):
                        for ch in "BGR":
                            data.append("BGR_MN:%d.%s.%s" % (i,axis,ch))

            if feature == 'shape_outline':
                n = getattr(args, 'k', 15)
                for i in range(1, n+1):
                    data.append("OUTLINE:%d.X" % i)
                    data.append("OUTLINE:%d.Y" % i)

            if feature == 'shape_360':
                step = getattr(args, 'step', 1)
                output_functions = getattr(args, 'output_functions', {'mean_sd': 1})
                for f_name, f_args in vars(output_functions).iteritems():
                    if f_name == 'mean_sd':
                        for i in range(0, 360, step):
                            data.append("360:%d.MN" % i)
                            data.append("360:%d.SD" % i)

                    if f_name == 'color_histograms':
                        for i in range(0, 360, step):
                            for cs, bins in vars(f_args).iteritems():
                                for j, color in enumerate(cs):
                                    for k in range(1, bins[j]+1):
                                        data.append("360:%d.%s:%d" % (i,color,k))

        # Write classification columns.
        try:
            out_prefix = self.config.data.dependent_prefix
        except:
            out_prefix = OUTPUT_PREFIX

        for i in range(1, n_out + 1):
            out.append("%s%d" % (out_prefix, i))

        return (data, out)

class BatchMakeTrainData(MakeTrainData):
    """Generate training data."""

    def __init__(self, config, base_path, db_path):
        """Constructor for training data generator.

        Expects a configurations object `config`, a path to the root
        directory of the photos, and a path to the database file `db_path`
        containing photo meta data.
        """
        super(BatchMakeTrainData, self).__init__(config, base_path, db_path)

        # Get the taxonomic hierarchy from the database.
        with session_scope(db_path) as (session, metadata):
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

        # Get the taxonomic hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("missing `classification.hierarchy`")

    def batch_export(self, target):
        """Batch export training data to directory `target`."""
        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Train an ANN for each path in the classification hierarchy.
        for filter_ in self.classification_hierarchy_filters(levels, self.taxon_hr):
            level = levels.index(filter_['class'])
            train_file = os.path.join(target, self.class_hr[level].train_file)
            config = self.class_hr[level]

            # Replace any placeholders in the paths.
            for key, val in filter_['where'].items():
                val = val if val is not None else '_'
                train_file = train_file.replace("__%s__" % key, val)

            # Generate and export the training data.
            logging.info("Classifying images on %s" % self.readable_filter(filter_))
            try:
                self.export(train_file, filter_, config)
            except nbc.FileExistsError as e:
                # Don't export if the file already exists.
                logging.warning("Skipping: %s" % e)

class MakeAnn(nbc.Common):
    """Train an artificial neural network."""

    def __init__(self, config, args=None):
        """Constructor for network trainer.

        Expects a configurations object `config`, and optionally the
        script arguments `args`.
        """
        super(MakeAnn, self).__init__(config)
        self.args = args

    def train(self, train_file, output_file, config=None, overwrite=False):
        """Train an artificial neural network.

        Loads training data from a CSV file `train_file`, trains a neural
        network `output_file` with training settings from the `config`
        object.
        """
        if not os.path.isfile(train_file):
            raise IOError("Cannot open %s (no such file)" % train_file)
        if not overwrite and os.path.isfile(output_file):
            raise nbc.FileExistsError("Output file %s already exists." % output_file)
        if config and not isinstance(config, nbc.Struct):
            raise ValueError("Expected an nbclassify.Struct instance for `config`")

        # Instantiate the ANN trainer.
        trainer = nbc.TrainANN()
        if config is not None:
            trainer.connection_rate = getattr(config, 'connection_rate', 1)
            trainer.hidden_layers = getattr(config, 'hidden_layers', 1)
            trainer.hidden_neurons = getattr(config, 'hidden_neurons', 8)
            trainer.learning_rate = getattr(config, 'learning_rate', 0.7)
            trainer.epochs = getattr(config, 'epochs', 100000)
            trainer.desired_error = getattr(config, 'error', 0.00001)
            trainer.training_algorithm = getattr(config, 'training_algorithm', 'TRAIN_RPROP')
            trainer.activation_function_hidden = getattr(config, 'activation_function_hidden', 'SIGMOID_STEPWISE')
            trainer.activation_function_output = getattr(config, 'activation_function_output', 'SIGMOID_STEPWISE')

        # These arguments overwrite parameters in the configurations file.
        if self.args:
            if self.args.epochs is not None:
                trainer.epochs = self.args.epochs
            if self.args.error is not None:
                trainer.desired_error = self.args.error

        trainer.iterations_between_reports = trainer.epochs / 100

        # Get the prefix for the classification columns.
        try:
            dependent_prefix = self.config.data.dependent_prefix
        except:
            dependent_prefix = OUTPUT_PREFIX

        try:
            train_data = nbc.TrainData()
            train_data.read_from_file(train_file, dependent_prefix)
        except ValueError as e:
            logging.error("Failed to process the training data: %s" % e)
            sys.exit(1)

        # Train the ANN.
        ann = trainer.train(train_data)
        ann.save(str(output_file))
        logging.info("Artificial neural network saved to %s" % output_file)
        error = trainer.test(train_data)
        logging.info("Mean Square Error on training data: %f" % error)

class BatchMakeAnn(MakeAnn):
    """Generate training data."""

    def __init__(self, config, db_path, args):
        """Constructor for training data generator.

        Expects a configurations object `config`, a path to the root
        directory of the photos, and a path to the database file `db_path`
        containing photo meta data.
        """
        super(BatchMakeAnn, self).__init__(config, args)

        # Get the taxonomic hierarchy from the database.
        with session_scope(db_path) as (session, metadata):
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

        # Get the taxonomic hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("missing `classification.hierarchy`")

    def batch_train(self, data_dir, target):
        """Batch train neural networks.

        Training data is obtained from the directory `data_dir` and the
        neural networks are saved to the directory `target`. Which training
        data to train on is set in the classification hierarchy of the
        configurations.
        """
        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Train an ANN for each path in the classification hierarchy.
        for filter_ in self.classification_hierarchy_filters(levels, self.taxon_hr):
            level = levels.index(filter_['class'])
            train_file = os.path.join(data_dir, self.class_hr[level].train_file)
            ann_file = os.path.join(target, self.class_hr[level].ann_file)
            config = None if 'ann' not in self.class_hr[level] else self.class_hr[level].ann

            # Replace any placeholders in the paths.
            for key, val in filter_['where'].items():
                val = val if val is not None else '_'
                train_file = train_file.replace("__%s__" % key, val)
                ann_file = ann_file.replace("__%s__" % key, val)

            # Train the ANN.
            logging.info("Training network `%s` with training data from `%s` ..." % (ann_file, train_file))
            try:
                self.train(train_file, ann_file, config, overwrite=False)
            except nbc.FileExistsError as e:
                # Don't train if the file already exists.
                logging.warning("Skipping: %s" % e)

class TestAnn(nbc.Common):
    """Test an artificial neural network."""

    def __init__(self, config):
        super(TestAnn, self).__init__(config)
        self.test_data = None
        self.ann = None
        self.re_photo_id = re.compile(r'([0-9]+)\.\w+$')
        self.classifications = {}
        self.classifications_expected = {}

    def test(self, ann_file, test_file):
        """Test an artificial neural network."""
        if not os.path.isfile(ann_file):
            raise IOError("Cannot open %s (no such file)" % ann_file)
        if not os.path.isfile(test_file):
            raise IOError("Cannot open %s (no such file)" % test_file)

        # Get the prefix for the classification columns.
        try:
            dependent_prefix = self.config.data.dependent_prefix
        except:
            dependent_prefix = OUTPUT_PREFIX

        self.ann = libfann.neural_net()
        self.ann.create_from_file(ann_file)

        self.test_data = nbc.TrainData()
        try:
            self.test_data.read_from_file(test_file, dependent_prefix)
        except IOError as e:
            logging.error("Failed to process the test data: %s" % e)
            exit(1)

        logging.info("Testing the neural network...")
        fann_test_data = libfann.training_data()
        fann_test_data.set_train_data(self.test_data.get_input(), self.test_data.get_output())

        self.ann.test_data(fann_test_data)

        mse = self.ann.get_MSE()
        logging.info("Mean Square Error on test data: %f" % mse)

    def export_results(self, filename, db_path, filter_, error=0.01):
        """Export the classification results to a TSV file.

        Export the test results to a tab separated file `filename`. The class
        name for a codeword is obtained from the database `db_path`, using the
        classification filter `filter_`. A bit in a codeword is considered on
        if the mean square error for a bit is less or equal to `error`.
        """
        if self.test_data is None:
            raise RuntimeError("Test data is not set")

        # Get the classification categories from the database.
        with session_scope(db_path) as (session, metadata):
            classes = self.get_classes_from_filter(session, metadata, filter_)
            classes = list(classes)

        # Get the codeword for each class.
        if not classes:
            raise RuntimeError("No classes found for filter `%s`" % filter_)
        codewords = self.get_codewords(classes)

        # Write results to file.
        with open(filename, 'w') as fh:
            # Write the header.
            fh.write( "%s\n" % "\t".join(['ID','Class','Classification','Match']) )

            total = 0
            correct = 0
            for label, input, output in self.test_data:
                total += 1
                row = []

                if label:
                    row.append(label)
                else:
                    row.append("")

                if len(codewords) != len(output):
                    raise ValueError("Codeword length ({0}) does not " \
                        "match output length ({1}). Is the classification " \
                        "filter correct?".format(len(codewords), len(output))
                    )

                class_expected = self.get_classification(codewords, output, error)
                class_expected = [class_ for mse,class_ in class_expected]
                assert len(class_expected) == 1, "The codeword for a class can only have one positive value"
                row.append(class_expected[0])

                codeword = self.ann.run(input)
                class_ann = self.get_classification(codewords, codeword, error)
                class_ann = [class_ for mse,class_ in class_ann]

                row.append(", ".join(class_ann))

                # Assume a match if the first items of the classifications match.
                if len(class_ann) > 0 and class_ann[0] == class_expected[0]:
                    row.append("+")
                    correct += 1
                else:
                    row.append("-")

                fh.write( "%s\n" % "\t".join(row) )

            # Calculate fraction correctly classified.
            fraction = float(correct) / total

            # Write correctly classified fraction.
            fh.write( "%s\n" % "\t".join(['','','',"%.3f" % fraction]) )

        print "Correctly classified: %.1f%%" % (fraction*100)
        print "Testing results written to %s" % filename

    def test_with_hierarchy(self, db_path, test_data_dir, ann_dir,
                            max_error=0.001):
        """Test each ANN in a classification hierarchy and export results.

        Returns a 2-tuple ``(total_classified, correctly_classified)``.
        """
        self.classifications = {}
        self.classifications_expected = {}

        # Get the taxonomic hierarchy from the database.
        with session_scope(db_path) as (session, metadata):
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

        # Get the taxonomic hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("Missing `classification.hierarchy`")

        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Get the prefix for the classification columns.
        try:
            dependent_prefix = self.config.data.dependent_prefix
        except:
            dependent_prefix = OUTPUT_PREFIX

        # Get the expected and recognized classification for each sample in
        # the test data.
        for filter_ in self.classification_hierarchy_filters(levels, self.taxon_hr):
            logging.info("Classifying on %s" % self.readable_filter(filter_))

            level_name = filter_['class']
            level_n = levels.index(level_name)
            level = self.class_hr[level_n]
            test_file = os.path.join(test_data_dir, level.test_file)
            ann_file = os.path.join(ann_dir, level.ann_file)

            # Set the maximum error for classification.
            try:
                max_error = level.max_error
            except:
                pass

            # Replace any placeholders in the paths.
            for key, val in filter_['where'].items():
                val = val if val is not None else '_'
                test_file = test_file.replace("__%s__" % key, val)
                ann_file = ann_file.replace("__%s__" % key, val)

            # Get the class names for this node in the taxonomic hierarchy.
            path = []
            for name in levels:
                try:
                    path.append(filter_['where'][name])
                except:
                    pass
            classes = self.get_childs_from_hierarchy(self.taxon_hr, path)

            # Get the codeword for each class.
            if not classes:
                raise RuntimeError("No classes found for filter `%s`" % filter_)
            codewords = self.get_codewords(classes)

            # Load the ANN.
            ann = libfann.neural_net()
            ann.create_from_file(str(ann_file))

            # Load the test data.
            test_data = nbc.TrainData()
            test_data.read_from_file(test_file, dependent_prefix)

            # Test each sample in the test data.
            for label, input, output in test_data:
                assert len(codewords) == len(output), \
                    "Codeword length {0} does not match output " \
                    "length {1}".format(len(codewords), len(output))

                # Obtain the photo ID from the label.
                if not label:
                    raise ValueError("Label for test sample not set")
                    row.append(label)
                try:
                    photo_id = self.re_photo_id.search(label).group(1)
                    photo_id = int(photo_id)
                except IndexError:
                    raise RuntimeError("Failed to obtain the photo ID from the sample label")

                # Set the expected class.
                class_expected = self.get_classification(codewords, output, max_error)
                class_expected = [class_ for mse,class_ in class_expected]

                assert len(class_expected) == 1, "The codeword for a class can only have one positive value"

                # Get the recognized class.
                codeword = ann.run(input)
                class_ann = self.get_classification(codewords, codeword, max_error)
                class_ann = [class_ for mse,class_ in class_ann]

                # Save the classification at each level.
                if level_n == 0:
                    self.classifications[photo_id] = {}
                    self.classifications_expected[photo_id] = {}
                self.classifications[photo_id][level_name] = class_ann
                self.classifications_expected[photo_id][level_name] = class_expected

    def export_hierarchy_results(self, filename):
        """Export classification results generated by :meth:`test_with_hierarchy`."""
        if not self.classifications or not self.classifications_expected:
            raise RuntimeError("Classifications not set")

        total = 0
        correct = 0

        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Export the results.
        with open(filename, 'w') as fh:
            # Write the header.
            row = ['ID']
            row.extend(["%s (exp.)" % level for level in levels])
            row.extend(["%s (class.)" % level for level in levels])
            row.append('Match')
            fh.write( "%s\n" % "\t".join(row) )

            # Write the results.
            for photo_id, class_exp in self.classifications_expected.iteritems():
                # Check if we have a match.
                match = True
                for level in levels:
                    if level not in class_exp:
                        continue
                    expected = class_exp[level][0]
                    if expected not in self.classifications[photo_id][level]:
                        match = False
                        break

                # Construct the results row.
                row = [str(photo_id)]

                # Expected classification.
                for level in levels:
                    class_ = class_exp.get(level)
                    class_ = '' if class_ is None else class_[0]
                    row.append(class_)

                # Accuired classification.
                for level in levels:
                    class_ = self.classifications[photo_id].get(level, [])
                    class_ = ", ".join(class_)
                    row.append(class_)

                # Match or not.
                if match:
                    row.append("+")
                    correct += 1
                else:
                    row.append("-")

                fh.write("%s\n" % "\t".join(row))

                total += 1

        return (total, correct)

class ImageClassifier(nbc.Common):
    """Classify an image."""

    def __init__(self, config, ann_file, db_file):
        super(ImageClassifier, self).__init__(config)
        self.set_ann(ann_file)

        # Get the classification categories from the database.
        with session_scope(db_file) as (session, metadata):
            try:
                filter_ = self.config.classification.filter
            except:
                raise nbc.ConfigurationError("Missing `classification.filter`")

            self.classes = self.get_classes_from_filter(session, metadata, filter_)
            self.classes = list(self.classes)

    def set_ann(self, path):
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        ann = libfann.neural_net()
        ann.create_from_file(path)
        self.ann = ann

    def classify(self, image_path, error=0.01):
        """Classify an image with a trained artificial neural network."""
        phenotyper = nbc.Phenotyper()
        phenotyper.set_image(image_path)
        phenotyper.set_config(self.config)
        phenotype = phenotyper.make()

        codeword = self.ann.run(phenotype)
        codewords = self.get_codewords(self.classes)
        return self.get_classification(codewords, codeword, error)

if __name__ == "__main__":
    main()
