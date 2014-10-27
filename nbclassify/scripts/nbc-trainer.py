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
from collections import Counter
import logging
import os
import re
import shelve
import sys

import cv2
import imgpheno as ft
import numpy as np
from pyfann import libfann
from sklearn import cross_validation
import sqlalchemy
import yaml

import nbclassify as nbc
import nbclassify.db as db

# Prefix for output columns in training data.
OUTPUT_PREFIX = "OUT:"

# File name of the meta data file.
META_FILE = ".meta.db"

# Force overwrite of files. If set to False, an nbclassify.FileExistsError is
# raised when an existing file is encountered. When set to True, any existing
# files are overwritten without warning.
FORCE_OVERWRITE = False

# Switch to True whilst debugging. It is automatically set to True when the
# -d switch is set on the Python interpreter. Setting this to True prevents
# some exceptions from being caught.
DEBUG = False

# Default settings.
ANN_DEFAULTS = {
    'connection_rate': 1,
    'hidden_layers': 1,
    'hidden_neurons': 8,
    'learning_rate': 0.7,
    'epochs': 100000,
    'desired_error': 0.00001,
    'training_algorithm': 'TRAIN_RPROP',
    'activation_function_hidden': 'SIGMOID_STEPWISE',
    'activation_function_output': 'SIGMOID_STEPWISE'
}

def main():
    global session, metadata, DEBUG

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
        DEBUG = True
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format='%(levelname)s %(message)s')

    # Get path to meta data file.
    try:
        meta_path = os.path.join(args.imdir, META_FILE)
    except:
        meta_path = None

    # Load the configurations.
    config = nbc.open_config(args.conf)

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
    except nbc.ConfigurationError as e:
        logging.error("A configurational error was detected: %s", e)
        return 1
    except nbc.FileExistsError as e:
        logging.error("An output file already exists: %s", e)
        return 1
    except Exception as e:
        if DEBUG: raise
        logging.error(e)
        return 1

    return 0

def meta(config, meta_path, args):
    """Make meta data file for an image directory."""
    global session, metadata

    sys.stdout.write("Initializing database...\n")
    db.make_meta_db(meta_path)

    with db.session_scope(meta_path) as (session, metadata):
        mkmeta = MakeMeta(config, args.imdir)
        mkmeta.make()

def data(config, meta_path, args):
    """Start train data routines."""
    global session, metadata

    try:
        filter_ = config.classification.filter.as_dict()
    except:
        raise nbc.ConfigurationError("The classification filter is not set")

    with db.session_scope(meta_path) as (session, metadata):
        cache = FingerprintCache(config)
        cache.make(args.imdir, args.cache_dir, update=False)

        train_data = MakeTrainData(config, args.cache_dir)
        train_data.export(args.output, filter_, config)

def data_batch(config, meta_path, args):
    """Start batch train data routines."""
    global session, metadata

    with db.session_scope(meta_path) as (session, metadata):
        cache = FingerprintCache(config)
        cache.make(args.imdir, args.cache_dir, update=False)

        train_data = BatchMakeTrainData(config, args.cache_dir)
        train_data.batch_export(args.output)

def ann(config, args):
    """Start neural network training routines."""
    ann_maker = MakeAnn(config)
    ann_maker.train(args.data, args.output)

def ann_batch(config, meta_path, args):
    """Start batch neural network training routines."""
    global session, metadata

    with db.session_scope(meta_path) as (session, metadata):
        ann_maker = BatchMakeAnn(config)
        ann_maker.batch_train(args.data, args.output)

def test_ann(config, meta_path, args):
    """Start neural network testing routines."""
    global session, metadata

    with db.session_scope(meta_path) as (session, metadata):
        tester = TestAnn(config)
        tester.test(args.ann, args.test_data)

        if args.output:
            try:
                filter_ = config.classification.filter.as_dict()
            except:
                raise nbc.ConfigurationError("The classification filter is not set")

            tester.export_results(args.output, filter_, args.error)

def test_ann_batch(config, meta_path, args):
    """Start batch neural network testing routines."""
    global session, metadata

    with db.session_scope(meta_path) as (session, metadata):
        tester = TestAnn(config)
        tester.test_with_hierarchy(args.test_data, args.anns, args.error)

        if args.output:
            correct, total = tester.export_hierarchy_results(args.output)
            print "Correctly classified: {0}/{1} ({2:.2%})\n".\
                format(correct, total, float(correct)/total)

def classify(config, meta_path, args):
    """Start classification routines."""
    global session, metadata

    with db.session_scope(meta_path) as (session, metadata):
        classifier = ImageClassifier(config, args.ann)
        classification = classifier.classify(args.image, args.error)

    class_ = [class_ for mse,class_ in classification]
    print "Image is classified as {0}".format(", ".join(class_))

def validate(config, meta_path, args):
    """Start validation routines."""
    global FORCE_OVERWRITE, session, metadata

    # Any existing training data or neural networks must be regenerated during
    # the validation process.
    FORCE_OVERWRITE = True

    with db.session_scope(meta_path) as (session, metadata):
        cache = FingerprintCache(config)
        cache.make(args.imdir, args.cache_dir, update=False)

        validator = Validator(config, args.cache_dir)
        scores = validator.k_fold_xval_stratified(args.k, args.autoskip)

    paths = (
        "genus",
        "genus/section",
        "genus/section/species",
    )

    print
    for path in paths:
        values = np.array(scores[path])

        print "Accuracy[{path}]: {mean:.2%} (+/- {sd:.2%})".format(**{
            'path': path,
            'mean': values.mean(),
            'sd': values.std() * 2
        })

class FingerprintCache(nbc.Common):

    """Cache an retrieve fingerprints."""

    def __init__(self, config):
        super(FingerprintCache, self).__init__(config)
        self._cache = {}

    def get_cache(self):
        """Return the cache as a nested dictionary."""
        return self._cache

    def combined_hash(self, *args):
        """Create a combined hash from one or more hashable objects.

        Each argument must be an hashable object. Can also be used for
        configuration objects as returned by :meth:`open_config`. Returned hash
        is a negative or positive integer.

        Example::

            >>> a = Struct({'a': True})
            >>> b = Struct({'b': False})
            >>> combined_hash(a,b)
            6862151379155462073
        """
        hash_ = None
        for obj in args:
            if hash_ is None:
                hash_ = hash(obj)
            else:
                hash_ ^= hash(obj)
        return hash_

    def make(self, image_dir, cache_dir, update=False):
        """Cache fingerprints to disk.

        One cache file is created per feature type to be extracted, which are
        stored in the target directory `cache_dir`. Each cache file is a Python
        shelve, a persistent, dictionary-like object. If `update` is set to
        True, existing fingerprints are updated.
        """
        global session, metadata

        phenotyper = nbc.Phenotyper()

        # Get a list of all the photos in the database.
        photos = db.get_photos(session, metadata)

        # Get the classification hierarchy.
        try:
            hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("classification hierarchy not set")

        # Cache each feature for each photo separately. One cache per
        # feature type is created, and each cache contains the fingerprints
        # for all images.
        hash_pp_ft = self.get_preprocess_feature_combinations(hr)
        for hash_, preprocess, feature in hash_pp_ft:
            cache_path = os.path.join(cache_dir, str(hash_))

            logging.info("Caching fingerprints in `{0}`...".format(cache_path))

            # Create a shelve for storing features. Empty existing shelves.
            cache = shelve.open(cache_path)

            # Cache the fingerprint for each photo.
            for photo in photos:
                # Skip fingerprinting if the fingerprint already exists, unless
                # update is set to True.
                if not update and str(photo.md5sum) in cache:
                    continue

                logging.info("Processing photo `%s`..." % photo.id)

                # Construct a new configuration object with a single feature
                # which we can pass to Phenotyper().
                config = nbc.Struct({
                    'preprocess': preprocess,
                    'features': feature
                })

                # Create a fingerprint and cache it.
                im_path = os.path.join(image_dir, photo.path)
                phenotyper.set_image(im_path)
                phenotyper.set_config(config)
                cache[str(photo.md5sum)] = phenotyper.make()

            cache.close()

    def get_preprocess_feature_combinations(self, hr):
        """Return preprocess/feature setting combinations from a classification
        hierarchy `hr`.

        This is a generator that returns 3-tuples ``(hash, preprocess,
        {feature_name: feature})``. The hash is unique for each returned
        preprocess/feature settings combination and can be recreated with
        :meth:`settings_hash`.
        """
        seen_hashes = []

        # Traverse the classification hierarchy for features that need to be
        # extracted.
        for level in hr:
            # Get the preprocess and features settings for this level.
            try:
                features = level.features
            except:
                raise nbc.ConfigurationError("features not set in " \
                    "classification hierarchy level")
            try:
                preprocess = level.preprocess
            except:
                preprocess = None

            for name, feature in vars(features).items():
                # Create a hash from the preprocessing and feature settings.
                # Preprocessing needs to be included because this also affects
                # the outcome of the features extracted. The hash must be
                # unique for each preprocessing/feature settings combination.
                hash_ = self.combined_hash(preprocess, feature)
                if hash_ not in seen_hashes:
                    seen_hashes.append(hash_)
                    yield (hash_, preprocess, {name: feature})

    def get_fingerprints(self, path, hash_):
        """Return fingerprints from cache for a given hash.

        Looks for caches in the directory `path` with the hash `hash_`. Returns
        None if the cache could not be found.
        """
        try:
            cache = shelve.open(os.path.join(path, str(hash_)))
            c = dict(cache)
            cache.close()
            return c
        except:
            return None

    def load_cache(self, cache_dir, config):
        """Load cache from `cache_dir` for preprocess/features settings in
        `config`.
        """
        try:
            features = config.features
        except:
            raise nbc.ConfigurationError("features not set")
        try:
            preprocess = config.preprocess
        except:
            preprocess = None

        self._cache = {}
        for f_name, f in vars(features).iteritems():
            hash_ = self.combined_hash(preprocess, f)
            cache = self.get_fingerprints(cache_dir, hash_)
            if cache is None:
                raise ValueError("Cache for hash {0} not found".format(hash_))
            self._cache[f_name] = cache

    def get_phenotype(self, key):
        """Return the phenotype for key `key`.

        The `key` is whatever was used as a key for storing the cache. Method
        :meth:`load_cache` must be called before calling this method.
        """
        if not self._cache:
            raise ValueError("Cache is not loaded")

        phenotype = []
        for k in sorted(self._cache.keys()):
            phenotype.extend(self._cache[k][str(key)])

        return phenotype

class MakeMeta(nbc.Common):

    """Create a meta data database for an image directory.

    The images in the image directory must be stored in a directory hierarchy
    which corresponds to the directory hierarchy set in the configurations.
    The meta data is created in the same directory. If a meta data file already
    exists, a FileExistsError is raised.
    """

    def __init__(self, config, image_dir):
        """Expects a configurations object `config` and a path to the directory
        containing the images `image_dir`.
        """
        super(MakeMeta, self).__init__(config)
        self.set_image_dir(image_dir)

        try:
            directory_hierarchy = list(config.directory_hierarchy)
        except:
            raise nbc.ConfigurationError("directory hierarchy is not set")

        # Set the ranks.
        self.ranks = []
        for rank in directory_hierarchy:
            if rank == "__ignore__":
                rank = None
            self.ranks.append(rank)

    def set_image_dir(self, path):
        """Set the image directory."""
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.image_dir = os.path.abspath(path)

    def get_image_files(self, root, ranks, classes=[]):
        """Return image paths and their classes.

        Images are returned as 2-tuples ``(path, {rank: class, ...})`` where
        each class is the directory name for each rank in `ranks`, a list of
        ranks. List `classes` is used internally to keep track of the classes.
        """
        if len(classes) > len(ranks):
            return

        for item in os.listdir(root):
            path = os.path.join(root, item)
            if os.path.isdir(path):
                # The current directory name is the class name.
                class_ = os.path.basename(path.strip(os.sep))
                if class_ in ("None", "NULL", "_"):
                    class_ = None
                for image in self.get_image_files(path, ranks, classes+[class_]):
                    yield image
            elif os.path.isfile(path) and classes:
                yield (path, dict(zip(ranks, classes)))

    def make(self):
        """Create the meta data database file `meta_path`."""
        global session, metadata

        sys.stdout.write("Saving meta data for images...\n")

        for path, classes in self.get_image_files(self.image_dir, self.ranks):
            # Get the path relative to self.image_dir
            path_rel = re.sub(self.image_dir, "", path)
            if path_rel.startswith(os.sep):
                path_rel = path_rel[1:]

            # Save the meta data.
            db.insert_new_photo(session, metadata,
                root=self.image_dir,
                path=path_rel,
                taxa=classes)

        sys.stdout.write("Done\n")

class MakeTrainData(nbc.Common):

    """Generate training data."""

    def __init__(self, config, cache_path):
        """Constructor for training data generator.

        Expects a configurations object `config`, and a path to the database
        file `meta_path` containing photo meta data.
        """
        super(MakeTrainData, self).__init__(config)
        self.set_cache_path(cache_path)
        self.subset = None
        self.cache = FingerprintCache(config)

    def set_cache_path(self, path):
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.cache_path = path

    def set_subset(self, subset):
        """Set the sample subset that should be used for export.

        A subset of the samples can be exported by providing a list of sample
        IDs `subset` for the samples to be exported. If subset is None, all
        samples are exported.
        """
        if subset is not None and not isinstance(subset, set):
            subset = set(subset)
        self.subset = subset

    def export(self, filename, filter_, config):
        """Write the training data to `filename`.

        Images to be processed are obtained from the database. Which images are
        obtained and with which classes is set by the filter `filter_`. Image
        fingerprints are obtained from cache, which must have been created for
        configuration `config`.
        """
        global session, metadata

        if not FORCE_OVERWRITE and os.path.isfile(filename):
            raise nbc.FileExistsError(filename)

        # Get the photos and corresponding classification using the filter.
        images = db.get_filtered_photos_with_taxon(session, metadata, filter_)
        images = images.all()

        if not images:
            logging.info("No images found for the filter `%s`" % filter_)
            return

        if self.get_photo_count_min():
            assert len(images) >= self.get_photo_count_min(), \
                "Expected to find at least photo_count_min={0} photos, found " \
                "{1}".format(self.get_photo_count_min(), len(images))

        # Calculate the number of images that will be processed, taking into
        # account the subset.
        photo_ids = np.array([photo.id for photo, _ in images])

        if self.subset:
            n_images = len(np.intersect1d(list(photo_ids), list(self.subset)))
        else:
            n_images = len(images)

        logging.info("Going to process %d photos..." % n_images)

        # Get the classification categories from the database.
        classes = self.get_classes_from_filter(session, metadata, filter_)

        # Make a codeword for each class.
        codewords = self.get_codewords(classes)

        # Construct the header.
        header_data, header_out = self.__make_header(len(classes))
        header = ["ID"] + header_data + header_out

        # Load the fingerprint cache.
        self.cache.load_cache(self.cache_path, config)

        # Generate the training data.
        with open(filename, 'w') as fh:
            # Write the header.
            fh.write( "%s\n" % "\t".join(header) )

            # Set the training data.
            training_data = nbc.TrainData(len(header_data), len(classes))

            for photo, class_ in images:
                # Only export the subset if an export subset is set.
                if self.subset and photo.id not in self.subset:
                    continue

                logging.info("Processing `%s` of class `%s`..." % (photo.path,
                    class_))

                # Get phenotype for this image from the cache.
                phenotype = self.cache.get_phenotype(photo.md5sum)

                assert len(phenotype) == len(header_data), \
                    "Fingerprint size mismatch. According to the header " \
                    "there are {0} data columns, but the fingerprint has " \
                    "{1}".format(len(header_data), len(phenotype))

                training_data.append(phenotype, codewords[class_],
                    label=photo.id)

            training_data.finalize()

            if not training_data:
                raise ValueError("Training data cannot be empty")

            # Round feature data.
            training_data.round_input(6)

            # Write data rows.
            for photo_id, input_, output in training_data:
                row = [str(photo_id)]
                row.extend(input_.astype(str))
                row.extend(output.astype(str))
                fh.write("%s\n" % "\t".join(row))

        logging.info("Training data written to %s" % filename)

    def __make_header(self, n_out):
        """Construct a header from features settings.

        Header is returned as a 2-tuple ``(data_columns, output_columns)``.
        """
        if 'features' not in self.config:
            raise nbc.ConfigurationError("missing `features`")

        data = []
        out = []

        # Always sort the features by name so that the headers match the
        # data column.
        features = sorted(vars(self.config.features).keys())

        for feature in features:
            args = self.config.features[feature]

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

    def __init__(self, config, cache_path):
        """Constructor for training data generator.

        Expects a configurations object `config`, a path to the root directory
        of the photos, and a path to the database file `meta_path` containing
        photo meta data.
        """
        super(BatchMakeTrainData, self).__init__(config, cache_path)

        self.taxon_hr = None

        # Set the classification hierarchy.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("classification hierarchy not set")

    def _load_taxon_hierarchy(self):
        global session, metadata

        if not self.taxon_hr:
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

    def batch_export(self, target_dir):
        """Batch export training data to directory `target_dir`."""
        # Must not be loaded in the constructor, in case set_photo_count_min()
        # is used.
        self._load_taxon_hierarchy()

        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Make training data for each path in the classification hierarchy.
        for filter_ in self.classification_hierarchy_filters(levels,
                self.taxon_hr):
            level = levels.index(filter_.get('class'))
            train_file = os.path.join(target_dir,
                self.class_hr[level].train_file)
            config = self.class_hr[level]

            # Replace any placeholders in the paths.
            where = filter_.get('where', {})
            for key, val in where.items():
                val = val if val is not None else '_'
                train_file = train_file.replace("__%s__" % key, val)

            # Generate and export the training data.
            logging.info("Classifying images on %s" % \
                self.readable_filter(filter_))
            try:
                self.export(train_file, filter_, config)
            except nbc.FileExistsError as e:
                # Don't export if the file already exists.
                logging.warning("Skipping: %s" % e)

class MakeAnn(nbc.Common):

    """Train an artificial neural network."""

    def __init__(self, config):
        """Constructor for network trainer.

        Expects a configurations object `config`, and optionally the
        script arguments `args`.
        """
        super(MakeAnn, self).__init__(config)

    def train(self, train_file, output_file, config=None):
        """Train an artificial neural network.

        Loads training data from a TSV file `train_file`, trains a neural
        network `output_file` with training settings from the configurations
        set with :meth:`set_config`. If training paramerters are provided with
        `config`, those are used instead.
        """
        if not os.path.isfile(train_file):
            raise IOError("Cannot open %s (no such file)" % train_file)
        if not FORCE_OVERWRITE and os.path.isfile(output_file):
            raise nbc.FileExistsError(output_file)
        if config and not isinstance(config, nbc.Struct):
            raise ValueError("Expected an nbclassify.Struct instance for `config`")

        # Instantiate the ANN trainer.
        trainer = nbc.TrainANN()

        # Set the training parameters.
        if not config:
            try:
                config = self.config.ann
            except:
                pass

        for option, value in ANN_DEFAULTS.iteritems():
            if config:
                value = getattr(config, option, value)
            setattr(trainer, option, value)

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

    def __init__(self, config):
        """Constructor for training data generator.

        Expects a configurations object `config`, a path to the root directory
        of the photos, and a path to the database file `meta_path` containing
        photo meta data.
        """
        super(BatchMakeAnn, self).__init__(config)

        self.taxon_hr = None

        # Get the taxonomic hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("missing `classification.hierarchy`")

    def _load_taxon_hierarchy(self):
        global session, metadata

        if not self.taxon_hr:
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

    def batch_train(self, data_dir, output_dir):
        """Batch train neural networks.

        Training data is obtained from the directory `data_dir` and the
        neural networks are saved to the directory `output_dir`. Which training
        data to train on is set in the classification hierarchy of the
        configurations.
        """
        # Must not be loaded in the constructor, in case set_photo_count_min()
        # is used.
        self._load_taxon_hierarchy()

        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Train an ANN for each path in the classification hierarchy.
        for filter_ in self.classification_hierarchy_filters(levels, self.taxon_hr):
            level = levels.index(filter_.get('class'))
            train_file = os.path.join(data_dir, self.class_hr[level].train_file)
            ann_file = os.path.join(output_dir, self.class_hr[level].ann_file)
            if 'ann' not in self.class_hr[level]:
                config = None
            else:
                config = self.class_hr[level].ann

            # Replace any placeholders in the paths.
            where = filter_.get('where', {})
            for key, val in where.items():
                val = val if val is not None else '_'
                train_file = train_file.replace("__%s__" % key, val)
                ann_file = ann_file.replace("__%s__" % key, val)

            # Train the ANN.
            logging.info("Training network `%s` with training data " \
                "from `%s` ..." % (ann_file, train_file))
            try:
                self.train(train_file, ann_file, config)
            except nbc.FileExistsError as e:
                # Don't train if the file already exists.
                logging.warning("Skipping: %s" % e)

class TestAnn(nbc.Common):

    """Test an artificial neural network."""

    def __init__(self, config):
        super(TestAnn, self).__init__(config)
        self.test_data = None
        self.ann = None
        self.re_photo_id = re.compile(r'([0-9]+)')
        self.classifications = {}
        self.classifications_expected = {}
        self.class_hr = None
        self.taxon_hr = None

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
        fann_test_data.set_train_data(self.test_data.get_input(),
            self.test_data.get_output())

        self.ann.test_data(fann_test_data)

        mse = self.ann.get_MSE()
        logging.info("Mean Square Error on test data: %f" % mse)

    def export_results(self, filename, filter_, error=0.01):
        """Export the classification results to a TSV file.

        Export the test results to a tab separated file `filename`. The class
        name for a codeword is obtained from the database `db_path`, using the
        classification filter `filter_`. A bit in a codeword is considered on
        if the mean square error for a bit is less or equal to `error`.
        """
        global session, metadata

        if self.test_data is None:
            raise RuntimeError("Test data is not set")

        # Get the classification categories from the database.
        classes = db.get_classes_from_filter(session, metadata, filter_)

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
                assert len(class_expected) == 1, \
                    "The codeword for a class can only have one positive value"
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

    def test_with_hierarchy(self, test_data_dir, ann_dir, max_error=0.001):
        """Test each ANN in a classification hierarchy and export results.

        Returns a 2-tuple ``(correct,total)``.
        """
        global session, metadata

        logging.info("Testing the neural networks hierarchy...")

        self.classifications = {}
        self.classifications_expected = {}

        # Get the taxonomic hierarchy from the database.
        self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

        # Get the classification hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("classification hierarchy not set")

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

            level_name = filter_.get('class')
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
            where = filter_.get('where', {})
            for key, val in where.items():
                val = val if val is not None else '_'
                test_file = test_file.replace("__%s__" % key, val)
                ann_file = ann_file.replace("__%s__" % key, val)

            # Get the class names for this node in the taxonomic hierarchy.
            path = self.path_from_filter(filter_, levels)
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
            for label, input_, output in test_data:
                assert len(codewords) == len(output), \
                    "Codeword size mismatch. Codeword has {0} bits, but the " \
                    "training data has {1} output bits.".\
                    format(len(codewords), len(output))

                # Obtain the photo ID from the label.
                if not label:
                    raise ValueError("Test sample is missing a label with " \
                        "photo ID")

                try:
                    photo_id = self.re_photo_id.search(label).group(1)
                    photo_id = int(photo_id)
                except:
                    raise RuntimeError("Failed to obtain the photo ID from " \
                        "the sample label")

                # Set the expected class.
                class_expected = self.get_classification(codewords, output,
                    max_error)
                class_expected = [class_ for mse,class_ in class_expected]

                assert len(class_expected) == 1, \
                    "Class codewords must have one positive bit, found {0}".\
                    format(len(class_expected))

                # Get the recognized class.
                codeword = ann.run(input_)
                class_ann = self.get_classification(codewords, codeword,
                    max_error)
                class_ann = [class_ for mse,class_ in class_ann]

                # Save the classification at each level.
                if level_n == 0:
                    self.classifications[photo_id] = {}
                    self.classifications_expected[photo_id] = {}

                self.classifications[photo_id][level_name] = class_ann
                self.classifications_expected[photo_id][level_name] = class_expected

            ann.destroy()

        return self.get_correct_count()

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

        return (correct, total)

    def get_correct_count(self, level_filter=None):
        """Return number of correctly classified samples.

        The classifications are checked on the level names in the list
        `level_filter`. If `level_filter` is None, all levels are checked. A
        classification is considered correct if the classifications at all
        levels are correct. If the classification for a level returns multiple
        values, the level is considered correct if the expected value is one
        of the members.

        Returns a dictinary where the keys correspond to the level name, and
        "all" for the overall score. Each value is a 2-tuples
        ``(correct,total)``.
        """
        if not self.classifications or not self.classifications_expected:
            raise RuntimeError("Classifications not set")

        total = len(self.classifications_expected)
        correct = 0
        levels = [l.name for l in self.class_hr]

        # Count the number of correct classifications.
        for photo_id, class_exp in self.classifications_expected.iteritems():
            match = True
            for level in levels:
                if level_filter and not level in level_filter:
                    continue

                try:
                    expected = class_exp[level][0]
                except:
                    continue

                if expected not in self.classifications[photo_id][level]:
                    match = False
                    break

            if match:
                correct += 1

        return (correct, total)

class ImageClassifier(nbc.Common):

    """Classify an image."""

    def __init__(self, config, ann_file):
        global session, metadata

        super(ImageClassifier, self).__init__(config)
        self.set_ann(ann_file)

        # Get the classification categories from the database.
        try:
            filter_ = self.config.classification.filter.as_dict()
        except:
            raise nbc.ConfigurationError("Missing `classification.filter`")

        self.classes = db.get_classes_from_filter(session, metadata, filter_)

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

class Validator(nbc.Common):

    """Validate artificial neural networks."""

    def __init__(self, config, cache_path):
        """Constructor for the validator.

        Expects a configurations object `config`, the path to to the directory
        with Flickr harvested images `base_path`, and the path to the meta data
        file `meta_path`.
        """
        super(Validator, self).__init__(config)
        self.classifications = {}
        self.classifications_expected = {}
        self.set_cache_path(cache_path)

    def set_cache_path(self, path):
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.cache_path = path

    def k_fold_xval_stratified(self, k=3, autoskip=False):
        """Perform stratified K-folds cross validation.

        The number of folds `k` must be at least 2. The minimum number of labels
        for any class cannot be less than `k`, or an AssertionError is raised.
        If `autoskip` is set to True, only the photos for species with at least
        `k` photos are used for the cross validation.
        """
        global session, metadata

        # Will hold the score of each folds.
        scores = {}

        # Get a list of all the photo IDs in the database.
        samples = db.get_photos_with_taxa(session, metadata)

        # Get a list of the photo IDs and a list of the classes. The classes
        # are needed for the stratified cross validation.
        photo_ids = []
        classes = []
        for x in samples:
            photo_ids.append(x[0].id)
            tmp = np.array(x[1:]).astype(str)
            classes.append('_'.join(tmp))

        # Numpy features are needed for these.
        photo_ids = np.array(photo_ids)
        classes = np.array(classes)

        # Count the number of each class.
        class_counts = Counter(classes)

        if autoskip:
            # Create a mask for the classes that have enough members and remove
            # the photo IDs that don't have enough members.
            mask = []
            for i, c in enumerate(classes):
                if class_counts[c] >= k:
                    mask.append(i)

            photo_ids = photo_ids[mask]
            classes = classes[mask]
        else:
            for label, count in class_counts.items():
                assert count >= k, "Class {0} has only {1} members, which " \
                    "is too few. The minimum number of labels for any " \
                    "class cannot be less than k={2}. Use --autoskip to skip " \
                    "classes with too few members.".format(label, count, k)

        if autoskip:
            photo_count_min = k
        else:
            photo_count_min = 0

        # Train data exporter.
        train_data = BatchMakeTrainData(self.config, self.cache_path)
        train_data.set_photo_count_min(photo_count_min)

        # Obtain cross validation folds.
        folds = cross_validation.StratifiedKFold(classes, k)
        result_dir = os.path.join(self.cache_path, 'results')
        for i, (train_idx, test_idx) in enumerate(folds):
            # Make data directories.
            train_dir = os.path.join(self.cache_path, 'train', str(i))
            test_dir = os.path.join(self.cache_path, 'test', str(i))
            ann_dir = os.path.join(self.cache_path, 'ann', str(i))
            test_result = os.path.join(result_dir, '{0}.tsv'.format(i))

            for path in (train_dir,test_dir,ann_dir,result_dir):
                if not os.path.isdir(path):
                    os.makedirs(path)

            # Make train data for this fold.
            train_samples = photo_ids[train_idx]
            train_data.set_subset(train_samples)
            train_data.batch_export(train_dir)

            # Make test data for this fold.
            test_samples = photo_ids[test_idx]
            train_data.set_subset(test_samples)
            train_data.batch_export(test_dir)

            # Train neural networks on training data.
            trainer = BatchMakeAnn(self.config)
            trainer.set_photo_count_min(photo_count_min)
            trainer.batch_train(data_dir=train_dir, output_dir=ann_dir)

            # Calculate the score for this fold.
            tester = TestAnn(self.config)
            tester.set_photo_count_min(photo_count_min)
            tester.test_with_hierarchy(test_dir, ann_dir)
            tester.export_hierarchy_results(test_result)

            level_filters = (
                ['genus'],
                ['genus','section'],
                ['genus','section','species']
            )

            for filter_ in level_filters:
                correct, total = tester.get_correct_count(filter_)
                score = float(correct) / total

                filter_s = "/".join(filter_)
                if filter_s not in scores:
                    scores[filter_s] = []
                scores[filter_s].append(score)

        return scores

if __name__ == "__main__":
    main()
