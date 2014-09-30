#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used to test the performance of trained artificial neural
networks of a classification hierarchy.

See the --help option for any of these subcommands for more information.
"""

import argparse
from contextlib import contextmanager
import hashlib
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

# Force overwrite files. This is used for testing.
FORCE_OVERWRITE = True

# File name of the meta data file.
META_FILE = ".meta.db"

# Prefix for output columns in training data.
OUTPUT_PREFIX = "OUT:"

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
    parser = argparse.ArgumentParser(
        description="Test the performance of trained artificial neural " \
        "networks."
    )

    parser.add_argument(
        "--conf",
        metavar="FILE",
        required=True,
        help="Path to a configurations file with the classification " \
        "hierarchy set.")
    parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        required=True,
        help="Path to a directory where temporary data and cache is stored.")
    parser.add_argument(
        "--error",
        metavar="N",
        type=float,
        default=0.0001,
        help="Significance threshold for mean square error values. " \
        "Defaults to 0.0001. This value can be overridden if max_error is " \
        "set in the classification hierarchy.")
    parser.add_argument(
        "--verbose",
        "-v",
        action='store_const',
        const=True,
        help="Explain what is being done.")
    parser.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where the Flickr harvested images are stored.")

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

    if not os.path.isfile(meta_path):
        logging.error("Directory not recognized as a Flickr harvested images " \
            "collection.")
        return 1

    config = nbc.open_config(args.conf)

    cache = FingerprintCache(config)
    cache.make(args.imdir, args.cache_dir, overwrite=False)

    validator = Validator(config, args.cache_dir, meta_path)
    scores = validator.k_fold_xval_stratified(3)

    sys.stderr.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),
        scores.std() * 2))

    return 0

@contextmanager
def session_scope(meta_path):
    """Provide a transactional scope around a series of operations."""
    engine = sqlalchemy.create_engine('sqlite:///%s' % os.path.abspath(meta_path),
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

    def make(self, image_dir, cache_dir, overwrite=False):
        """Cache fingerprints to disk.

        One cache file is created per feature type to be extracted, which are
        stored in the target directory `cache_dir`. Each cache file is a Python
        shelve, a persistent, dictionary-like object.
        """
        phenotyper = nbc.Phenotyper()
        meta_path = os.path.join(image_dir, META_FILE)

        # Get a list of all the photos in the database.
        with session_scope(meta_path) as (session, metadata):
            images = self.get_photos(session, metadata)
            images = list(images)

        # Get the classification hierarchy.
        try:
            hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("missing `classification.hierarchy`")

        # Cache each feature for each photo separately. One cache per
        # feature type is created, and each cache contains the fingerprints
        # for all images.
        hash_pp_ft = self.get_preprocess_feature_combinations(hr)
        for hash_, preprocess, feature in hash_pp_ft:
            cache_path = os.path.join(cache_dir, str(hash_))

            if not overwrite and os.path.isfile(cache_path):
                logging.info("Cache `{0}` exists. Skipping."
                    .format(cache_path))
                continue

            logging.info("Collecting fingerprints for cache `{0}`..."
                .format(cache_path))

            # Create a shelve for storing features. Empty existing shelves.
            cache = shelve.open(cache_path)
            cache.clear()

            # Cache the fingerprint for each photo.
            for im_id, im_path in images:
                logging.info("Processing photo `%s`..." % im_id)

                # Construct a new configuration object with a single feature
                # which we can pass to Phenotyper().
                config = nbc.Struct({
                    'preprocess': preprocess,
                    'features': feature
                })

                # Create a fingerprint and cache it.
                im_path = os.path.join(image_dir, im_path)
                phenotyper.set_image(im_path)
                phenotyper.set_config(config)
                cache[str(im_id)] = phenotyper.make()

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
                raise nbc.ConfigurationError("missing `features` in " \
                    "classification hierarchy level")
            try:
                preprocess = level.preprocess
            except:
                preprocess = None

            for name, feature in vars(features).items():
                # Create a hash from the proprocessing and feature settings.
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
            raise nbc.ConfigurationError("missing `features`")
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

    def get_phenotype_for_photo(self, photo_id):
        """Return the phenotype for a photo with ID `photo_id`.

        Method :meth:`load_cache` must be called before calling this method.
        """
        if not self._cache:
            raise ValueError("Cache is not loaded")

        phenotype = []
        for k in sorted(self._cache.keys()):
            phenotype.extend(self._cache[k][str(photo_id)])

        return phenotype

class Validator(nbc.Common):

    """Validate artificial neural networks."""

    def __init__(self, config, cache_path, meta_path):
        """Constructor for the validator.

        Expects a configurations object `config`, the path to to the directory
        with Flickr harvested images `base_path`, and the path to the meta data
        file `meta_path`.
        """
        super(Validator, self).__init__(config)
        self.classifications = {}
        self.classifications_expected = {}
        self.set_cache_path(cache_path)
        self.set_meta_path(meta_path)

    def set_cache_path(self, path):
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.cache_path = path

    def set_meta_path(self, path):
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        self.meta_path = path

    def k_fold_xval_stratified(self, k=3):
        """Perform stratified K-folds cross validation.

        Arguments are the `samples` to split in K folds, and the number of folds
        `k`, which must be at least 2.
        """
        # Will hold the score of each folds.
        scores = []

        # Get a list of all the photo IDs in the database.
        with session_scope(self.meta_path) as (session, metadata):
            samples = self.get_photo_ids(session, metadata)
            samples = np.array(list(samples))

        # Train data exporter.
        train_data = BatchMakeTrainData(self.config, self.cache_path,
            self.meta_path)

        # Obtain cross validation folds.
        folds = cross_validation.StratifiedKFold(samples, k, shuffle=True)
        for i, (train_idx, test_idx) in enumerate(folds):
            # Make data directories.
            train_dir = os.path.join(self.cache_path, 'train', str(i))
            test_dir = os.path.join(self.cache_path, 'test', str(i))

            if not os.path.isdir(train_dir):
                os.makedirs(train_dir)
            if not os.path.isdir(test_dir):
                os.makedirs(test_dir)

            # Make train data for this fold.
            train_samples = samples[train_idx]
            train_data.set_subset(train_samples)
            train_data.batch_export(train_dir)

            # Make test data for this fold.
            test_samples = samples[test_idx]
            train_data.set_subset(test_samples)
            train_data.batch_export(test_dir)

            # Train neural networks on training data.
            trainer = BatchMakeAnn(self.config, self.meta_path)
            trainer.batch_train(data_dir=train_dir, output_dir=train_dir)

            # Calculate the score for this fold.
            tester = TestAnn(self.config)
            correct, total = tester.test_with_hierarchy(self.meta_path,
                test_dir, train_dir)
            score = float(correct) / total
            scores.append(score)

        return np.array(scores)

class MakeTrainData(nbc.Common):

    """Generate training data."""

    def __init__(self, config, cache_path, meta_path):
        """Constructor for training data generator.

        Expects a configurations object `config`, and a path to the database
        file `meta_path` containing photo meta data.
        """
        super(MakeTrainData, self).__init__(config)
        self.set_cache_path(cache_path)
        self.set_meta_path(meta_path)
        self.subset = None
        self.cache = FingerprintCache(config)

    def set_cache_path(self, path):
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.cache_path = path

    def set_meta_path(self, path):
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        self.meta_path = path

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
        if not FORCE_OVERWRITE and os.path.isfile(filename):
            raise nbc.FileExistsError("Output file %s already exists." % filename)

        # Get list of image paths and corresponding classifications from the
        # meta data database.
        with session_scope(self.meta_path) as (session, metadata):
            images = self.get_photos_with_class(session, metadata, filter_)
            images = np.array(list(images))

        if len(images) == 0:
            logging.info("No images found for the filter `%s`" % filter_)
            return

        logging.info("Going to process %d photos..." % len(images))

        # Get the classification categories from the database.
        with session_scope(self.meta_path) as (session, metadata):
            classes = self.get_classes_from_filter(session, metadata, filter_)

        # Make a codeword for each class.
        codewords = self.get_codewords(classes)

        # Construct the header.
        header_data, header_out = self.__make_header(len(classes))
        header = ["ID"] + header_data + header_out

        # Open the fingerprint cache.
        self.cache.load_cache(self.cache_path, config)

        # Generate the training data.
        with open(filename, 'w') as fh:
            # Write the header.
            fh.write( "%s\n" % "\t".join(header) )

            # Set the training data.
            training_data = nbc.TrainData(len(header_data), len(classes))

            for photo_id, photo_path, photo_class in images:
                # Only export the subset if an export subset is set.
                if self.subset and int(photo_id) not in self.subset:
                    continue

                logging.info("Processing `%s` of class `%s`..." % (photo_path,
                    photo_class))

                # Create a phenotype from the image.
                phenotype = self.cache.get_phenotype_for_photo(photo_id)

                assert len(phenotype) == len(header_data), \
                    "Fingerprint length mismatch"

                training_data.append(phenotype, codewords[photo_class],
                    label=photo_id)

            training_data.finalize()

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

    def __init__(self, config, cache_path, meta_path):
        """Constructor for training data generator.

        Expects a configurations object `config`, a path to the root
        directory of the photos, and a path to the database file `meta_path`
        containing photo meta data.
        """
        super(BatchMakeTrainData, self).__init__(config, cache_path, meta_path)

        # Get the taxonomic hierarchy from the database.
        with session_scope(meta_path) as (session, metadata):
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

        # Get the classification hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("missing `classification.hierarchy`")

    def batch_export(self, target_dir):
        """Batch export training data to directory `target_dir`."""
        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Make training data for each path in the classification hierarchy.
        for filter_ in self.classification_hierarchy_filters(levels, self.taxon_hr):
            level = levels.index(filter_['class'])
            train_file = os.path.join(target_dir, self.class_hr[level].train_file)
            config = self.class_hr[level]

            # Replace any placeholders in the paths.
            for key, val in filter_['where'].items():
                val = val if val is not None else '_'
                train_file = train_file.replace("__%s__" % key, val)

            # Generate and export the training data.
            logging.info("Classifying images on %s" % self.readable_filter(filter_))
            self.export(train_file, filter_, config)

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

        Loads training data from a CSV file `train_file`, trains a neural
        network `output_file` with training settings from the `config`
        object.
        """
        if config and not isinstance(config, nbc.Struct):
            raise ValueError("Expected an nbclassify.Struct instance for `config`")

        # Instantiate the ANN trainer.
        trainer = nbc.TrainANN()
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

    def __init__(self, config, meta_path):
        """Constructor for training data generator.

        Expects a configurations object `config`, a path to the root
        directory of the photos, and a path to the database file `meta_path`
        containing photo meta data.
        """
        super(BatchMakeAnn, self).__init__(config)

        # Get the taxonomic hierarchy from the database.
        with session_scope(meta_path) as (session, metadata):
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

        # Get the taxonomic hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("missing `classification.hierarchy`")

    def batch_train(self, data_dir, output_dir):
        """Batch train neural networks.

        Training data is obtained from the directory `data_dir` and the
        neural networks are saved to the directory `output_dir`. Which training
        data to train on is set in the classification hierarchy of the
        configurations.
        """
        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Train an ANN for each path in the classification hierarchy.
        for filter_ in self.classification_hierarchy_filters(levels, self.taxon_hr):
            level = levels.index(filter_['class'])
            train_file = os.path.join(data_dir, self.class_hr[level].train_file)
            ann_file = os.path.join(output_dir, self.class_hr[level].ann_file)
            config = None if 'ann' not in self.class_hr[level] else self.class_hr[level].ann

            # Replace any placeholders in the paths.
            for key, val in filter_['where'].items():
                val = val if val is not None else '_'
                train_file = train_file.replace("__%s__" % key, val)
                ann_file = ann_file.replace("__%s__" % key, val)

            # Train the ANN.
            logging.info("Training network `%s` with training data from `%s` ..." % (ann_file, train_file))
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
        fann_test_data.set_train_data(self.test_data.get_input(), self.test_data.get_output())

        self.ann.test_data(fann_test_data)

        mse = self.ann.get_MSE()
        logging.info("Mean Square Error on test data: %f" % mse)

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
            for label, input_, output in test_data:
                assert len(codewords) == len(output), \
                    "Codeword length {0} does not match output " \
                    "length {1}".format(len(codewords), len(output))

                # Obtain the photo ID from the label.
                if not label:
                    raise ValueError("Label for test sample not set")

                try:
                    photo_id = self.re_photo_id.search(label).group(1)
                    photo_id = int(photo_id)
                except IndexError:
                    raise RuntimeError("Failed to obtain the photo ID from " \
                        "the sample label")

                # Set the expected class.
                class_expected = self.get_classification(codewords, output,
                    max_error)
                class_expected = [class_ for mse,class_ in class_expected]

                assert len(class_expected) == 1, \
                    "The codeword for a class can only have one positive value"

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

        return self.get_correct_count()

    def get_correct_count(self):
        """Return number of correctly classified samples.

        Returns a 2-tuple ``(correct,total)``.
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

if __name__ == "__main__":
    main()
