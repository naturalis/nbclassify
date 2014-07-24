#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trainer for the artificial neural networks.

Tasks:

* data: Create a tab separated file with training data.
* ann: Train an artificial neural network.
* test-ann: Test the performance of an artificial neural network.
* classify: Classify an image using an artificial neural network.
"""

import argparse
from contextlib import contextmanager
import logging
import os
import sys

import cv2
import numpy as np
from pyfann import libfann
import sqlalchemy
import sqlalchemy.orm as orm
from sqlalchemy.ext.automap import automap_base
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
    subparsers = parser.add_subparsers(help="Specify which task to start.")

    # Create an argument parser for sub-command 'data'.
    help_data = """Create a tab separated file with training data.
    Preprocessing steps and features to extract must be set in a
    configurations file. See trainer.yml for an example."""

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

    # Create an argument parser for sub-command 'batch-data'.
    help_batch_data = """Batch create tab separated files with training
    data. Preprocessing steps and features to extract must be set in a
    configurations file. See trainer.yml for an example."""

    parser_batch_data = subparsers.add_parser('batch-data',
        help=help_batch_data, description=help_batch_data)
    parser_batch_data.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a YAML file with feature extraction parameters.")
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
    parameters can be set in a separate configurations file. See
    trainer.yml for an example file."""

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

    # Create an argument parser for sub-command 'batch-ann'.
    help_batch_ann = """Batch train artificial neural networks. Optional training
    parameters can be set in a separate configurations file. See
    trainer.yml for an example file."""

    parser_batch_ann = subparsers.add_parser('batch-ann',
        help=help_batch_ann, description=help_batch_ann)
    parser_batch_ann.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a YAML file with ANN training parameters.")
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
    set, then `--conf` must also be set. See trainer.yml for an example YAML
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

        config = open_yaml(args.conf)

        try:
            filter_ = config.classification.filter
        except:
            logging.error("The configuration file is missing object classification.filter")
            return

        try:
            train_data = MakeTrainData(config, args.basepath, args.db)
            train_data.export(args.output, filter_, config)
        except Exception as e:
            logging.error(e)
            raise

    if sys.argv[1] == 'batch-data':
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

    elif sys.argv[1] == 'ann':
        try:
            config = open_yaml(args.conf)
            ann_maker = MakeAnn(config, args)
            ann_maker.train(args.data, args.output)
        except Exception as e:
            logging.error(e)

    elif sys.argv[1] == 'batch-ann':
        try:
            config = open_yaml(args.conf)
            ann_maker = BatchMakeAnn(config, args.db, args)
            ann_maker.batch_train(args.data, args.output)
        except Exception as e:
            logging.error(e)
            raise

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
    return nbc.Struct(d)

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
        if not isinstance(config, nbc.Struct):
            raise TypeError("Configurations object must be of type nbc.Struct, not %s" % type(config))

        try:
            path = config.preprocess.segmentation.output_folder
        except:
            path = None
        if path and not os.path.isdir(path):
            logging.error("Found a configuration error")
            raise IOError("Cannot open %s (no such directory)" % path)

        self.config = config

    def get_codewords(self, classes, on=1, off=-1):
        """Return codewords for a list of classes."""
        n =  len(classes)
        codewords = {}
        for i, class_ in enumerate(sorted(classes)):
            cw = [off] * n
            cw[i] = on
            codewords[class_] = cw
        return codewords

    def get_classification(self, codewords, codeword, error=0.01):
        """Return a human-readable classification from a codeword."""
        if len(codewords) != len(codeword):
            raise ValueError("Lenth of `codewords` must be equal to `codeword`")
        classes = []
        for class_, cw in codewords.items():
            for i, code in enumerate(cw):
                if code == 1.0 and (code - codeword[i])**2 <= error:
                    classes.append((codeword[i], class_))
        classes = [x[1] for x in sorted(classes, reverse=True)]
        return classes

    def query_images(self, session, metadata, filter_):
        """Return photos with corresponding class from the database.

        Photos obtained from the database are filtered by rules set in the
        `filter_` parameter. Returned rows are (photo_path, class) tuples.
        """
        if 'class' not in filter_:
            raise ValueError("The filter is missing the 'class' key")
        if isinstance(filter_, dict):
            filter_ = nbc.Struct(filter_)
        for key in vars(filter_):
            if key not in ('where', 'class'):
                raise ValueError("Unknown key '%s' in filter" % key)

        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = Base.classes.photos_taxa
        Taxa = Base.classes.taxa
        Rank = Base.classes.ranks

        # Construct the sub queries.
        stmt1 = session.query(PhotosTaxa.photo_id, Taxa.name.label('genus')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'genus').subquery()
        stmt2 = session.query(PhotosTaxa.photo_id, Taxa.name.label('section')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'section').subquery()
        stmt3 = session.query(PhotosTaxa.photo_id, Taxa.name.label('species')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'species').subquery()

        # Construct the query.
        q = session.query(Photos.path, getattr(filter_, 'class')).\
            join(stmt1).\
            outerjoin(stmt2).\
            join(stmt3)

        # Add the WHERE clauses to the query.
        if 'where' in filter_:
            for rank, taxon in vars(filter_.where).items():
                if rank == 'genus':
                    q = q.filter(stmt1.c.genus == taxon)
                elif rank == 'section':
                    q = q.filter(stmt2.c.section == taxon)
                elif rank == 'species':
                    q = q.filter(stmt3.c.species == taxon)

        return q

    def query_classes(self, session, metadata, filter_):
        """Return classes from the database.

        Returned classes are filtered by rules set in the `filter_` parameter.
        Returned rows are (photo_id, class) tuples.
        """
        if 'class' not in filter_:
            raise ValueError("The filter is missing the 'class' key")
        if isinstance(filter_, dict):
            filter_ = nbc.Struct(filter_)
        for key in vars(filter_):
            if key not in ('where', 'class'):
                raise ValueError("Unknown key '%s' in filter" % key)

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

    def get_taxon_hierarchy(self, session, metadata):
        """Return the taxanomic hierarchy for photos in the database.

        The hierarchy is returned as a dictionary in the format
        ``{genus: {section: [species, ..], ..}, ..}``.
        """
        hierarchy = {}
        q = self.get_taxa(session, metadata)

        for _, genus, section, species in q:
            if genus not in hierarchy:
                hierarchy[genus] = {}
            if section not in hierarchy[genus]:
                hierarchy[genus][section] = []
            hierarchy[genus][section].append(species)
        return hierarchy

    def get_taxa(self, session, metadata):
        """Return the taxa from the database.

        Taxa are returned as (genus, section, species) tuples.
        """
        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = Base.classes.photos_taxa
        Taxa = Base.classes.taxa
        Rank = Base.classes.ranks

        # Construct the sub queries.
        stmt1 = session.query(PhotosTaxa.photo_id, Taxa.name.label('genus')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'genus').subquery()
        stmt2 = session.query(PhotosTaxa.photo_id, Taxa.name.label('section')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'section').subquery()
        stmt3 = session.query(PhotosTaxa.photo_id, Taxa.name.label('species')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'species').subquery()

        # Construct the query.
        q = session.query(Photos.id, 'genus', 'section', 'species').\
            join(stmt1).\
            outerjoin(stmt2).\
            join(stmt3).\
            group_by('genus', 'section', 'species')

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

    def export(self, filename, filter_, config):
        """Write the training data to `filename`.

        Images to be processed are obtained from the database. Which images
        are obtained and with which classes is set by the filter
        `filter_`. A configuration object `config` denotes which
        preprocessing steps to take and what features are to be extracted.
        """
        if os.path.isfile(filename):
            logging.error("Output file `%s` already exists. Please choose a different file name." % filename)
            return

        # Get list of image paths and corresponding classifications from
        # the meta data database.
        with session_scope(self.db_path) as (session, metadata):
            q = self.query_images(session, metadata, filter_)
            images = list(q)

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
            self.taxon_hierarchy = self.get_taxon_hierarchy(session, metadata)

    def batch_export(self, target):
        """Batch export training data to directory `target`."""
        try:
            classify_hierarchy = self.config.classification.hierarchy
        except:
            raise ValueError("The configuration file is missing object classification.hierarchy")

        # Classify on genus level.
        level = 0
        data_file = os.path.join(target, classify_hierarchy[level].data)
        if not os.path.isfile(data_file):
            logging.info("Generating training data for the filter `%s` ..." % filter_)
            self.export(data_file, {'class': 'genus'}, classify_hierarchy[level])
        else:
            logging.warning("File `%s` already exists. Skipping." % data_file)

        for genus, sections in self.taxon_hierarchy.items():
            if sections.keys() == [None]:
                logging.info("No sections for genus `%s`" % genus)
            else:
                filter_ = {
                    'where': {
                        'genus': genus
                    },
                    'class': 'section'
                }

                # Classify on section level.
                level = 1
                data_file = os.path.join(target, classify_hierarchy[level].data)
                data_file = data_file.replace("__genus__", genus)
                if not os.path.isfile(data_file):
                    logging.info("Generating training data for the filter `%s` ..." % filter_)
                    self.export(data_file, filter_, classify_hierarchy[level])
                else:
                    logging.warning("File `%s` already exists. Skipping." % data_file)

            for section, species in sections.items():
                filter_ = {
                    'where': {
                        'genus': genus,
                        'section': section
                    },
                    'class': 'species'
                }

                # Classify on species level.
                level = 2
                if section is None: section = '_'
                data_file = os.path.join(target, classify_hierarchy[level].data)
                data_file = data_file.replace("__genus__", genus)
                data_file = data_file.replace("__section__", section)
                if not os.path.isfile(data_file):
                    logging.info("Generating training data for the filter `%s` ..." % filter_)
                    self.export(data_file, filter_, classify_hierarchy[level])
                else:
                    logging.warning("File `%s` already exists. Skipping." % data_file)

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
        if os.path.isfile(output_file):
            logging.error("Output file %s already exists. Please choose a different file name." % output_file)
            return

        # Instantiate the ANN trainer.
        trainer = nbc.TrainANN()

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
            self.taxon_hierarchy = self.get_taxon_hierarchy(session, metadata)

    def batch_train(self, data_dir, target):
        """Batch train neural networks into directory `target`."""
        try:
            classify_hierarchy = self.config.classification.hierarchy
        except:
            raise ValueError("The configuration file is missing object classification.hierarchy")

        # Train on genus level.
        level = 0
        data_file = os.path.join(data_dir, classify_hierarchy[level].data)
        ann_file = os.path.join(target, classify_hierarchy[level].ann)
        if not os.path.isfile(ann_file):
            logging.info("Training network with training data from `%s` ..." % data_file)
            try:
                self.train(data_file, ann_file)
            except Exception as e:
                logging.error("Failed to train network: %s" % e)
        else:
            logging.warning("File `%s` already exists. Skipping." % ann_file)

        for genus, sections in self.taxon_hierarchy.items():
            if sections.keys() == [None]:
                logging.info("No sections for genus `%s`" % genus)
            else:
                # Train on section level.
                level = 1
                data_file = os.path.join(data_dir, classify_hierarchy[level].data)
                data_file = data_file.replace("__genus__", genus)
                ann_file = os.path.join(target, classify_hierarchy[level].ann)
                ann_file = ann_file.replace("__genus__", genus)
                if not os.path.isfile(ann_file):
                    logging.info("Training network with training data from `%s` ..." % data_file)
                    try:
                        self.train(data_file, ann_file)
                    except Exception as e:
                        logging.error("Failed to train network: %s" % e)
                else:
                    logging.warning("File `%s` already exists. Skipping." % ann_file)

            for section, species in sections.items():
                # Train on species level.
                level = 2
                if section is None: section = '_'
                data_file = os.path.join(data_dir, classify_hierarchy[level].data)
                data_file = data_file.replace("__genus__", genus)
                data_file = data_file.replace("__section__", section)
                ann_file = os.path.join(target, classify_hierarchy[level].ann)
                ann_file = ann_file.replace("__genus__", genus)
                ann_file = ann_file.replace("__section__", section)
                if not os.path.isfile(ann_file):
                    logging.info("Training network with training data from `%s` ..." % data_file)
                    try:
                        self.train(data_file, ann_file)
                    except Exception as e:
                        logging.error("Failed to train network: %s" % e)
                else:
                    logging.warning("File `%s` already exists. Skipping." % ann_file)

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

    def export_results(self, filename, db_path, error=0.01):
        """Export the classification results to a TSV file."""
        if self.test_data is None:
            raise RuntimeError("Test data is not set")

        # Get the classification categories from the database.
        with session_scope(db_path) as (session, metadata):
            try:
                filter_ = self.config.classification.filter
            except:
                raise RuntimeError("The configuration file is missing object classification.filter")

            q = self.query_classes(session, metadata, filter_)
            classes = [x[1] for x in q]

        if len(classes) == 0:
            raise RuntimeError("No classes found for filter `%s`" % filter_)

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
                filter_ = self.config.classification.filter
            except:
                raise RuntimeError("The configuration file is missing object classification.filter")

            q = self.query_classes(session, metadata, filter_)
            self.classes = [x[1] for x in q]

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
