#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trainer for the artificial neural networks.

The following classes are defined:

* Phenotyper: Create phenotypes from an image.

Tasks:

* data: Create a tab separated file with training data.
* ann: Train an artificial neural network.
* test-ann: Test the performance of an artificial neural network.
* classify: Classify an image using an artificial neural network.
"""

import argparse
from contextlib import contextmanager
import csv
import logging
import mimetypes
import os
import sys

import cv2
import numpy as np
from pyfann import libfann
import sqlalchemy
import sqlalchemy.orm as orm
from sqlalchemy.ext.automap import automap_base
import yaml

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
    subparsers = parser.add_subparsers(help="Specify which task to start.")

    # Create an argument parser for sub-command 'data'.
    help_data = """Create a tab separated file with training data.
    Preprocessing steps and features to extract must be set in a YAML file.
    See trainer.yml for an example."""

    parser_data = subparsers.add_parser('data',
        help=help_data, description=help_data)
    parser_data.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a YAML file with feature extraction parameters.")
    parser_data.add_argument("--db", metavar="DB",
        help="Path to a database file with photo meta data. If omitted " \
        "this defaults to a file photos.db in the photo's directory.")
    parser_data.add_argument("--output", "-o", metavar="FILE", required=True,
        help="Output file name for training data. Any existing file with " \
        "same name will be overwritten.")
    parser_data.add_argument("basepath", metavar="PATH",
        help="Base directory where to look for photo's. The database file" \
        "with photo meta data will be used to find photo's in this directory.")

    # Create an argument parser for sub-command 'ann'.
    help_ann = """Train an artificial neural network. Optional training
    parameters can be set in a separate YAML file. See orchids.yml
    for an example file."""

    parser_ann = subparsers.add_parser('ann',
        help=help_ann, description=help_ann)
    parser_ann.add_argument("--conf", metavar="FILE",
        help="Path to a YAML file with ANN training parameters.")
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

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann = """Test an artificial neural network. If `--output` is
    set, then `--conf` must also be set. See orchids.yml for an example YAML
    file with class names."""

    parser_test_ann = subparsers.add_parser('test-ann',
        help=help_test_ann, description=help_test_ann)
    parser_test_ann.add_argument("--ann", metavar="FILE", required=True,
        help="A trained artificial neural network.")
    parser_test_ann.add_argument("--db", metavar="DB",
        help="Path to a database file with photo meta data. Must be" \
        "used together with --output.")
    parser_test_ann.add_argument("--output", "-o", metavar="FILE",
        help="Output file name for the test results. Specifying this " \
        "option will output a table with the classification result for " \
        "each sample.")
    parser_test_ann.add_argument("--conf", metavar="FILE",
        help="Path to a YAML file with class names.")
    parser_test_ann.add_argument("--error", metavar="N", type=float,
        default=0.00001,
        help="The maximum mean square error for classification. Default " \
        "is 0.00001")
    parser_test_ann.add_argument("data", metavar="TEST_DATA",
        help="Path to tab separated file containing test data.")

    # Create an argument parser for sub-command 'classify'.
    help_classify = """Classify an image. See orchids.yml for an example YAML
    file with class names."""

    parser_classify = subparsers.add_parser('classify',
        help=help_classify, description=help_classify)
    parser_classify.add_argument("--ann", metavar="FILE", required=True,
        help="Path to a trained artificial neural network file.")
    parser_classify.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a YAML file with class names.")
    parser_classify.add_argument("--db", metavar="DB", required=True,
        help="Path to a database file with photo meta data.")
    parser_classify.add_argument("--error", metavar="N", type=float,
        default=0.00001,
        help="The maximum error for classification. Default is 0.00001")
    parser_classify.add_argument("image", metavar="IMAGE",
        help="Path to image file to be classified.")

    # Parse arguments.
    args = parser.parse_args()

    if sys.argv[1] == 'data':
        # Set the default database path if not set.
        if args.db is None:
            args.db = os.path.join(args.basepath, 'photos.db')

        try:
            config = open_yaml(args.conf)
            train_data = MakeTrainData(config, args.basepath, args.db)
            train_data.export(args.output)
        except Exception as e:
            logging.error(e)
            raise

    elif sys.argv[1] == 'ann':
        try:
            config = open_yaml(args.conf)
            ann_maker = MakeAnn(config, args)
            ann_maker.train(args.data, args.output)
        except Exception as e:
            logging.error(e)

    elif sys.argv[1] == 'test-ann':
        config = open_yaml(args.conf)
        tester = TestAnn(config)
        tester.test(args.ann, args.data)
        if args.output:
            if not args.db:
                sys.exit("Option --output must be used together with --db")
            tester.export_results(args.output, args.db, args.error)

    elif sys.argv[1] == 'classify':
        config = open_yaml(args.conf)
        classifier = ImageClassifier(config, args.ann, args.db)
        class_ = classifier.classify(args.image, args.error)
        logging.info("Image is classified as %s" % ", ".join(class_))

    sys.exit()

@contextmanager
def session_scope(db_path):
    """Provide a transactional scope around a series of operations."""
    engine = sqlalchemy.create_engine('sqlite:///%s' % os.path.abspath(db_path),
        echo=sys.flags.debug)
    Session = orm.sessionmaker(bind=engine)
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
    return Struct(d)

def open_yaml(path):
    """Read a YAML file and return as an object."""
    with open(path, 'r') as f:
        config = yaml.load(f)
    return get_object(config)

class Common(object):
    def __init__(self, config):
        self.set_config(config)

    def set_config(self, config):
        """Set the YAML configurations object."""
        if not isinstance(config, Struct):
            raise TypeError("Configurations object must be of type Struct, not %s" % type(config))

        try:
            path = config.preprocess.segmentation.output_folder
        except:
            path = None
        if path and not os.path.isdir(path):
            logging.error("Found a configuration error")
            raise IOError("Cannot open %s (no such directory)" % path)

        self.config = config

    def get_codewords(self, classes, neg=-1, pos=1):
        """Return codewords for a list of classes."""
        n =  len(classes)
        codewords = {}
        for i, class_ in enumerate(sorted(classes)):
            cw = [neg] * n
            cw[i] = pos
            codewords[class_] = cw
        return codewords

    def get_classification(self, codewords, codeword, error=0.01):
        """Return a human-readable classification from a codeword."""
        if len(codewords) != len(codeword):
            raise ValueError("Lenth of `codewords` must be equal to `codeword`")
        classes = []
        for cls, cw in codewords.items():
            for i, code in enumerate(cw):
                if code == 1.0 and (code - codeword[i])**2 < error:
                    classes.append((codeword[i], cls))
        classes = [x[1] for x in sorted(classes, reverse=True)]
        return classes

    def query_images_classes(self, session, metadata, query):
        """Construct a query from the `class_query` parameter."""
        if 'class' not in query:
            raise ValueError("The query is missing the 'class' key")
        for key in vars(query):
            if key not in ('where', 'class'):
                raise ValueError("Unknown key '%s' in query" % key)

        # Poduce a set of mappings from the MetaData.
        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = {'class': Base.classes.photos_taxa}
        Taxa = {'class': Base.classes.taxa}
        Ranks = {'class': Base.classes.ranks}

        # Construct the query, ORM style.
        q = session.query(Photos.path, Taxa['class'].name)

        if 'where' in query:
            for rank, name in vars(query.where).items():
                PhotosTaxa[rank] = orm.aliased(Base.classes.photos_taxa)
                Taxa[rank] = orm.aliased(Base.classes.taxa)
                Ranks[rank] = orm.aliased(Base.classes.ranks)

                q = q.join(PhotosTaxa[rank], PhotosTaxa[rank].photo_id == Photos.id).\
                    join(Taxa[rank]).join(Ranks[rank]).\
                    filter(Ranks[rank].name == rank, Taxa[rank].name == name)

        # The classification column.
        rank = getattr(query, 'class')
        q = q.join(PhotosTaxa['class'], PhotosTaxa['class'].photo_id == Photos.id).\
            join(Taxa['class']).join(Ranks['class']).\
            filter(Ranks['class'].name == rank)

        # Order by classification.
        q = q.order_by(Taxa['class'].name)

        return q

    def query_classes(self, session, metadata, query):
        """Construct a query from the `class_query` parameter.

        The query selects a unique list of classification names.
        """
        if 'class' not in query:
            raise ValueError("The query is missing the 'class' key")
        for key in vars(query):
            if key not in ('where', 'class'):
                raise ValueError("Unknown key '%s' in query" % key)

        # Poduce a set of mappings from the MetaData.
        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = {'class': Base.classes.photos_taxa}
        Taxa = {'class': Base.classes.taxa}
        Ranks = {'class': Base.classes.ranks}

        # Construct the query, ORM style.
        q = session.query(Photos.id, Taxa['class'].name)

        if 'where' in query:
            for rank, name in vars(query.where).items():
                PhotosTaxa[rank] = orm.aliased(Base.classes.photos_taxa)
                Taxa[rank] = orm.aliased(Base.classes.taxa)
                Ranks[rank] = orm.aliased(Base.classes.ranks)

                q = q.join(PhotosTaxa[rank], PhotosTaxa[rank].photo_id == Photos.id).\
                    join(Taxa[rank]).join(Ranks[rank]).\
                    filter(Ranks[rank].name == rank, Taxa[rank].name == name)

        # The classification column.
        rank = getattr(query, 'class')
        q = q.join(PhotosTaxa['class'], PhotosTaxa['class'].photo_id == Photos.id).\
            join(Taxa['class']).join(Ranks['class']).\
            filter(Ranks['class'].name == rank)

        # Order by classification.
        q = q.group_by(Taxa['class'].name)

        return q

class MakeTrainData(Common):
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

    def export(self, filename):
        """Write the training data to `filename`."""

        # Get list of image paths and corresponding classifications from
        # the meta data database.
        with session_scope(self.db_path) as (session, metadata):
            try:
                class_query = self.config.class_query
            except:
                raise RuntimeError("Classification query not set in the configuration file. Option 'class_query' is missing.")

            q = self.query_images_classes(session, metadata, class_query)
            images = list(q)

        if len(images) == 0:
            logging.info("No images found for the query %s" % class_query)
            return

        logging.info("Going to process %d photos..." % len(images))

        # Get a unique list of classes.
        classes = set([x[1] for x in images])

        # Make codeword for each class.
        codewords = self.get_codewords(classes, -1, 1)

        # Construct the header.
        header_data, header_out = self.__make_header(len(classes))
        header = ["ID"] + header_data + header_out

        # Generate the training data.
        with open(filename, 'w') as fh:
            # Write the header.
            fh.write( "%s\n" % "\t".join(header) )

            # Set the training data.
            training_data = TrainData(len(header_data), len(classes))
            phenotyper = Phenotyper()
            failed = []
            for im_path, im_class in images:
                im_path_real = os.path.join(self.base_path, im_path)

                try:
                    phenotyper.set_image(im_path_real)
                    phenotyper.set_config(self.config)
                except:
                    logging.warning("Failed to read %s. Skipping." % im_path_real)
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

        Header is returned as a tuple (data_columns, output_columns).
        """
        if 'features' not in self.config:
            raise ValueError("Features not set in configuration")

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

class MakeAnn(Common):
    """Train an artificial neural network."""

    def __init__(self, config, args=None):
        """Constructor for network trainer.

        Expects a configurations object `config`, and optionally the
        script arguments `args`.
        """
        super(MakeAnn, self).__init__(config)
        self.args = args

    def train(self, train_file, output_file):
        """Train an artificial neural network."""
        if not os.path.isfile(train_file):
            raise IOError("Cannot open %s (no such file)" % train_file)

        # Instantiate the ANN trainer.
        trainer = TrainANN()

        if 'ann' in self.config:
            trainer.connection_rate = getattr(self.config.ann, 'connection_rate', 1)
            trainer.hidden_layers = getattr(self.config.ann, 'hidden_layers', 1)
            trainer.hidden_neurons = getattr(self.config.ann, 'hidden_neurons', 8)
            trainer.learning_rate = getattr(self.config.ann, 'learning_rate', 0.7)
            trainer.epochs = getattr(self.config.ann, 'epochs', 100000)
            trainer.desired_error = getattr(self.config.ann, 'error', 0.00001)
            trainer.training_algorithm = getattr(self.config.ann, 'training_algorithm', 'TRAIN_RPROP')
            trainer.activation_function_hidden = getattr(self.config.ann, 'activation_function_hidden', 'SIGMOID_STEPWISE')
            trainer.activation_function_output = getattr(self.config.ann, 'activation_function_output', 'SIGMOID_STEPWISE')

        # These arguments overwrite parameters in the configurations file.
        if self.args:
            if self.args.epochs != None:
                trainer.epochs = self.args.epochs
            if self.args.error != None:
                trainer.desired_error = self.args.error

        trainer.iterations_between_reports = trainer.epochs / 100

        # Get the prefix for the classification columns.
        try:
            dependent_prefix = self.config.data.dependent_prefix
        except:
            dependent_prefix = OUTPUT_PREFIX

        try:
            train_data = TrainData()
            train_data.read_from_file(train_file, dependent_prefix)
        except ValueError as e:
            logging.error("Failed to process the training data: %s" % e)
            sys.exit(1)

        # Train the ANN.
        ann = trainer.train(train_data)
        ann.save(output_file)
        logging.info("Artificial neural network saved to %s" % output_file)
        error = trainer.test(train_data)
        logging.info("Mean Square Error on training data: %f" % error)

class TestAnn(Common):
    """Test an artificial neural network."""

    def __init__(self, config):
        super(TestAnn, self).__init__(config)
        self.test_data = None
        self.ann = None

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

        self.test_data = TrainData()
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

    def export_results(self, filename, db_path, error=0.01):
        """Export the classification results to a TSV file."""
        if self.test_data is None:
            raise RuntimeError("Test data is not set")

        # Get the classification categories from the database.
        with session_scope(db_path) as (session, metadata):
            try:
                class_query = self.config.class_query
            except:
                raise RuntimeError("Classification query not set in the configuration file. Option 'class_query' is missing.")

            q = self.query_classes(session, metadata, class_query)
            classes = [x[1] for x in q]

        if len(classes) == 0:
            raise RuntimeError("No classes found for query %s" % class_query)

        with open(filename, 'w') as fh:
            fh.write( "%s\n" % "\t".join(['ID','Class','Classification','Match']) )

            codewords = self.get_codewords(classes)
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
                    raise RuntimeError("Codeword length (%d) does not match the " \
                        "output length (%d). Please make sure the test data " \
                        "matches the class query in the configurations " \
                        "file." % (len(codewords), len(output))
                    )

                class_expected = self.get_classification(codewords, output, error)
                assert len(class_expected) == 1, "The codeword for a class can only have one positive value"
                row.append(class_expected[0])

                codeword = self.ann.run(input)
                class_ann = self.get_classification(codewords, codeword, error)
                row.append(", ".join(class_ann))

                # Assume a match if the first items of the classifications match.
                if len(class_ann) > 0 and class_ann[0] == class_expected[0]:
                    row.append("+")
                    correct += 1
                else:
                    row.append("-")

                fh.write( "%s\n" % "\t".join(row) )

                fraction = float(correct) / total

            # Write correctly classified fraction.
            fh.write( "%s\n" % "\t".join(['','','',"%.3f" % fraction]) )

        logging.info("Correctly classified: %.1f%%" % (fraction*100))
        logging.info("Testing results written to %s" % filename)

class ImageClassifier(Common):
    """Classify an image."""

    def __init__(self, config, ann_file, db_file):
        super(ImageClassifier, self).__init__(config)
        self.set_ann(ann_file)

        # Get the classification categories from the database.
        with session_scope(db_file) as (session, metadata):
            try:
                class_query = self.config.class_query
            except:
                raise RuntimeError("Classification query not set in the configuration file. Option 'class_query' is missing.")

            q = self.query_classes(session, metadata, class_query)
            self.classes = [x[1] for x in q]

    def set_ann(self, path):
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        ann = libfann.neural_net()
        ann.create_from_file(path)
        self.ann = ann

    def classify(self, image_path, error=0.01):
        """Classify an image with a trained artificial neural network."""
        phenotyper = Phenotyper()
        phenotyper.set_image(image_path)
        phenotyper.set_config(self.config)
        phenotype = phenotyper.make()

        codeword = self.ann.run(phenotype)
        codewords = self.get_codewords(self.classes)
        return self.get_classification(codewords, codeword, error)

class Phenotyper(object):
    """Generate numerical features from an image."""

    def __init__(self):
        self.path = None
        self.config = None
        self.img = None
        self.mask = None
        self.bin_mask = None

    def set_image(self, path):
        self.img = cv2.imread(path)
        if self.img == None or self.img.size == 0:
            raise IOError("Failed to read image %s" % path)

        self.path = path
        self.config = None
        self.mask = None
        self.bin_mask = None

        return self.img

    def set_config(self, config):
        """Set the YAML configurations object."""
        if not isinstance(config, Struct):
            raise TypeError("Configurations object must be of type Struct, not %s" % type(config))
        self.config = config

    def __preprocess(self):
        if self.img is None:
            raise RuntimeError("No image is loaded")

        if 'preprocess' not in self.config:
            return

        # Scale the image down if its perimeter exceeds the maximum (if set).
        perim = sum(self.img.shape[:2])
        max_perim = getattr(self.config.preprocess, 'maximum_perimeter', None)
        if max_perim and perim > max_perim:
            logging.info("Scaling down...")
            rf = float(max_perim) / perim
            self.img = cv2.resize(self.img, None, fx=rf, fy=rf)

        # Perform color enhancement.
        color_enhancement = getattr(self.config.preprocess, 'color_enhancement', None)
        if color_enhancement:
            for method, args in vars(color_enhancement).iteritems():
                if method == 'naik_murthy_linear':
                    logging.info("Color enhancement...")
                    self.img = ft.naik_murthy_linear(self.img)
                else:
                    raise ValueError("Unknown color enhancement method '%s'" % method)

        # Perform segmentation.
        segmentation = getattr(self.config.preprocess, 'segmentation', None)
        if segmentation:
            logging.info("Segmenting...")
            iterations = getattr(segmentation, 'iterations', 5)
            margin = getattr(segmentation, 'margin', 1)
            output_folder = getattr(segmentation, 'output_folder', None)

            # Create a binary mask for the largest contour.
            self.mask = ft.segment(self.img, iterations, margin)
            self.bin_mask = np.where((self.mask==cv2.GC_FGD) + (self.mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
            contour = ft.get_largest_contour(self.bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contour == None:
                raise ValueError("No contour found for binary image")
            self.bin_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            cv2.drawContours(self.bin_mask, [contour], 0, 255, -1)

            # Save the masked image to the output folder.
            if output_folder and os.path.isdir(output_folder):
                img_masked = cv2.bitwise_and(self.img, self.img, mask=self.bin_mask)
                fname = os.path.basename(self.path)
                out_path = os.path.join(output_folder, fname)
                cv2.imwrite(out_path, img_masked)

    def make(self):
        if self.img == None:
            raise ValueError("No image loaded")

        logging.info("Processing %s ..." % self.path)

        self.__preprocess()

        logging.info("Extracting features...")

        data_row = []

        if not 'features' in self.config:
            raise RuntimeError("Features to extract not set. Nothing to do.")

        for feature, args in vars(self.config.features).iteritems():
            if feature == 'color_histograms':
                logging.info("- Running color:histograms...")
                data = self.get_color_histograms(self.img, args, self.bin_mask)
                data_row.extend(data)

            elif feature == 'color_bgr_means':
                logging.info("- Running color:bgr_means...")
                data = self.get_color_bgr_means(self.img, args, self.bin_mask)
                data_row.extend(data)

            elif feature == 'shape_outline':
                logging.info("- Running shape:outline...")
                data = self.get_shape_outline(args, self.bin_mask)
                data_row.extend(data)

            elif feature == 'shape_360':
                logging.info("- Running shape:360...")
                data = self.get_shape_360(args, self.bin_mask)
                data_row.extend(data)

            else:
                raise ValueError("Unknown feature '%s'" % feature)

        return data_row

    def get_color_histograms(self, src, args, bin_mask=None):
        histograms = []
        for colorspace, bins in vars(args).iteritems():
            if colorspace.lower() == "bgr":
                colorspace = ft.CS_BGR
                img = src
            elif colorspace.lower() == "hsv":
                colorspace = ft.CS_HSV
                img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            elif colorspace.lower() == "luv":
                colorspace = ft.CS_LUV
                img = cv2.cvtColor(src, cv2.COLOR_BGR2LUV)
            else:
                raise ValueError("Unknown colorspace '%s'" % colorspace)

            hists = ft.color_histograms(img, bins, bin_mask, colorspace)

            for hist in hists:
                hist = cv2.normalize(hist, None, -1, 1, cv2.NORM_MINMAX)
                histograms.extend( hist.ravel() )
        return histograms

    def get_color_bgr_means(self, src, args, bin_mask=None):
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        # Get the contours from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Create a masked image.
        img = cv2.bitwise_and(src, src, mask=bin_mask)

        bins = getattr(args, 'bins', 20)
        output = ft.color_bgr_means(img, contour, bins)

        # Normalize data to range -1 .. 1
        return output * 2.0 / 255 - 1

    def get_shape_outline(self, args, bin_mask):
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        k = getattr(args, 'k', 15)

        # Obtain contours (all points) from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Get the outline.
        outline = ft.shape_outline(contour, k)

        # Compute the delta's for the horizontal and vertical point pairs.
        shape = []
        for x, y in outline:
            delta_x = x[0] - x[1]
            delta_y = y[0] - y[1]
            shape.append(delta_x)
            shape.append(delta_y)

        # Normalize results.
        shape = np.array(shape, dtype=np.float32)
        shape = cv2.normalize(shape, None, -1, 1, cv2.NORM_MINMAX)

        return shape.ravel()

    def get_shape_360(self, args, bin_mask):
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        rotation = getattr(args, 'rotation', 0)
        step = getattr(args, 'step', 1)
        t = getattr(args, 't', 8)
        output_functions = getattr(args, 'output_functions', {'mean_sd': True})

        # Get the largest contour from the binary mask.
        contour = ft.get_largest_contour(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Set the rotation.
        if rotation == 'FIT_ELLIPSE':
            box = cv2.fitEllipse(contour)
            rotation = int(box[2])
        if not 0 <= rotation <= 179:
            raise ValueError("Rotation must be in the range 0 to 179, found %s" % rotation)

        # Extract shape feature.
        intersects, center = ft.shape_360(contour, rotation, step, t)

        # Create a masked image.
        if 'color_histograms' in output_functions:
            img_masked = cv2.bitwise_and(self.img, self.img, mask=bin_mask)

        # Run the output function for each angle.
        means = []
        sds = []
        histograms = []
        for angle in range(0, 360, step):
            for f_name, f_args in vars(output_functions).iteritems():
                # Mean distance + standard deviation.
                if f_name == 'mean_sd':
                    distances = []
                    for p in intersects[angle]:
                        d = ft.point_dist(center, p)
                        distances.append(d)

                    if len(distances) == 0:
                        mean = 0
                        sd = 0
                    else:
                        mean = np.mean(distances, dtype=np.float32)
                        if len(distances) > 1:
                            sd = np.std(distances, ddof=1, dtype=np.float32)
                        else:
                            sd = 0

                    means.append(mean)
                    sds.append(sd)

                # Color histograms.
                if f_name == 'color_histograms':
                    # Get a line from the center to the outer intersection point.
                    line = None
                    if len(intersects[angle]) > 0:
                        line = ft.extreme_points([center] + intersects[angle])

                    # Create a mask for the line, where the line is foreground.
                    line_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                    if line != None:
                        cv2.line(line_mask, tuple(line[0]), tuple(line[1]), 255, 1)

                    # Create histogram from masked + line masked image.
                    hists = self.get_color_histograms(img_masked, f_args, line_mask)
                    histograms.append(hists)

        # Normalize results.
        if 'mean_sd' in output_functions:
            means = cv2.normalize(np.array(means), None, -1, 1, cv2.NORM_MINMAX)
            sds = cv2.normalize(np.array(sds), None, -1, 1, cv2.NORM_MINMAX)

        # Group the means+sds together.
        means_sds = np.array(zip(means, sds)).flatten()

        return np.append(means_sds, histograms)

class Struct(argparse.Namespace):
    """Return a dictionary as an object."""

    def __init__(self, d):
        for key, val in d.iteritems():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [self.__class__(x) if \
                    isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, self.__class__(val) if \
                    isinstance(val, dict) else val)

class TrainData(object):
    """Class for storing training data."""

    def __init__(self, num_input = 0, num_output = 0):
        self.num_input = num_input
        self.num_output = num_output
        self.labels = []
        self.input = []
        self.output = []
        self.counter = 0

    def read_from_file(self, path, dependent_prefix="OUT:"):
        """Reads training data from file.

        Data is loaded from TSV file `path`. File must have a header row,
        and columns with a name starting with `dependent_prefix` are used as
        classification columns. Optionally, sample labels can be stored in
        a column with name "ID". All remaining columns are used as predictors.
        """
        with open(path, 'r') as fh:
            reader = csv.reader(fh, delimiter="\t")

            # Figure out the format of the data.
            header = reader.next()
            input_start = None
            output_start = None
            label_idx = None
            for i, field in enumerate(header):
                if field == "ID":
                    label_idx = i
                elif field.startswith(dependent_prefix):
                    if output_start == None:
                        output_start = i
                    self.num_output += 1
                else:
                    if input_start == None:
                        input_start = i
                    self.num_input += 1

            if self.num_input == 0:
                raise IOError("No input columns found in training data.")
            if self.num_output  == 0:
                raise IOError("No output columns found in training data.")

            input_end = input_start + self.num_input
            output_end = output_start + self.num_output

            for row in reader:
                if label_idx != None:
                    self.labels.append(row[label_idx])
                else:
                    self.labels.append(None)
                self.input.append(row[input_start:input_end])
                self.output.append(row[output_start:output_end])

            self.finalize()

    def __len__(self):
        return len(self.input)

    def __iter__(self):
        return self

    def next(self):
        if self.counter >= len(self.input):
            self.counter = 0
            raise StopIteration
        else:
            self.counter += 1
            i = self.counter - 1
            return (self.labels[i], self.input[i], self.output[i])

    def append(self, input, output, label=None):
        if isinstance(self.input, np.ndarray):
            raise ValueError("Cannot add data once finalized")
        if len(input) != self.num_input:
            raise ValueError("Incorrect input array length (expected length of %d)" % self.num_input)
        if len(output) != self.num_output:
            raise ValueError("Incorrect output array length (expected length of %d)" % self.num_output)

        self.labels.append(label)
        self.input.append(input)
        self.output.append(output)

    def finalize(self):
        self.input = np.array(self.input).astype(float)
        self.output = np.array(self.output).astype(float)

    def normalize_input_columns(self, alpha, beta, norm_type=cv2.NORM_MINMAX):
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this function")

        for col in range(self.num_input):
            tmp = cv2.normalize(self.input[:,col], None, alpha, beta, norm_type)
            self.input[:,col] = tmp[:,0]

    def normalize_input_rows(self, alpha, beta, norm_type=cv2.NORM_MINMAX):
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this function")

        for i, row in enumerate(self.input):
            self.input[i] = cv2.normalize(row, None, alpha, beta, norm_type).reshape(-1)

    def round_input(self, decimals=4):
        self.input = np.around(self.input, decimals)

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

class TrainANN(object):
    """Train an artificial neural network."""

    def __init__(self):
        self.ann = None
        self.connection_rate = 1
        self.learning_rate = 0.7
        self.hidden_layers = 1
        self.hidden_neurons = 8
        self.epochs = 500000
        self.iterations_between_reports = self.epochs / 100
        self.desired_error = 0.0001
        self.training_algorithm = 'TRAIN_RPROP'
        self.activation_function_hidden = 'SIGMOID_STEPWISE'
        self.activation_function_output = 'SIGMOID_STEPWISE'
        self.train_data = None
        self.test_data = None

    def set_train_data(self, data):
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        self.train_data = data

    def set_test_data(self, data):
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        if data.num_input != self.train_data.num_input:
            raise ValueError("Number of inputs of test data must be same as train data")
        if data.num_output != self.train_data.num_output:
            raise ValueError("Number of output of test data must be same as train data")
        self.test_data = data

    def train(self, train_data):
        self.set_train_data(train_data)

        hidden_layers = [self.hidden_neurons] * self.hidden_layers
        layers = [self.train_data.num_input]
        layers.extend(hidden_layers)
        layers.append(self.train_data.num_output)

        sys.stderr.write("Network layout:\n")
        sys.stderr.write("* Neuron layers: %s\n" % layers)
        sys.stderr.write("* Connection rate: %s\n" % self.connection_rate)
        if self.training_algorithm not in ('FANN_TRAIN_RPROP',):
            sys.stderr.write("* Learning rate: %s\n" % self.learning_rate)
        sys.stderr.write("* Activation function for the hidden layers: %s\n" % self.activation_function_hidden)
        sys.stderr.write("* Activation function for the output layer: %s\n" % self.activation_function_output)
        sys.stderr.write("* Training algorithm: %s\n" % self.training_algorithm)

        self.ann = libfann.neural_net()
        self.ann.create_sparse_array(self.connection_rate, layers)
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_activation_function_hidden(getattr(libfann, self.activation_function_hidden))
        self.ann.set_activation_function_output(getattr(libfann, self.activation_function_output))
        self.ann.set_training_algorithm(getattr(libfann, self.training_algorithm))

        fann_train_data = libfann.training_data()
        fann_train_data.set_train_data(self.train_data.get_input(), self.train_data.get_output())

        self.ann.train_on_data(fann_train_data, self.epochs, self.iterations_between_reports, self.desired_error)
        return self.ann

    def test(self, test_data):
        self.set_test_data(test_data)

        fann_test_data = libfann.training_data()
        fann_test_data.set_train_data(self.test_data.get_input(), self.test_data.get_output())

        self.ann.reset_MSE()
        self.ann.test_data(fann_test_data)

        return self.ann.get_MSE()


if __name__ == "__main__":
    main()
