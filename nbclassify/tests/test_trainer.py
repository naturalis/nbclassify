#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for the training and classification routines.

Tests whether the routines run without raising unexpected exceptions. Outputs
and results are not checked here.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

from context import *
from nbclassify import conf, open_config
from nbclassify.classify import ImageClassifier
from nbclassify.data import PhenotypeCache, MakeTrainData, BatchMakeTrainData
from nbclassify.functions import (delete_temp_dir, get_classification,
    get_codewords)
from nbclassify.training import MakeAnn, TestAnn, BatchMakeAnn
from nbclassify.validate import Validator
import nbclassify.db as db

META_FILE = os.path.join(IMAGE_DIR, conf.meta_file)

# Disable FileExistsError exceptions.
conf.force_overwrite = True

# Raise exceptions which would otherwise be caught.
conf.debug = True

def validate(config, k, autoskip=False):
    """Start validation routines."""
    with db.session_scope(META_FILE) as (session, metadata):
        cache = PhenotypeCache()
        cache.make(IMAGE_DIR, TEMP_DIR, config, update=False)

        validator = Validator(config, TEMP_DIR)
        scores = validator.k_fold_xval_stratified(k, autoskip)

    print
    for path in sorted(scores.keys()):
        values = np.array(scores[path])

        print "Accuracy[{path}]: {mean:.2%} (+/- {sd:.2%})".format(**{
            'path': path,
            'mean': values.mean(),
            'sd': values.std() * 2
        })

#@unittest.skip("Debugging")
class TestTrainer(unittest.TestCase):

    """Test the trainer script.

    This simply tests if the trainer script runs without crashing unexpectedly.
    It does not test any output data. It is important that the tests run in a
    fixed order, which is why the test functions are named alphabetically.
    """

    @classmethod
    def setUpClass(cls):
        """Clean up the temporary directory and remove an existing metadata
        file.

        This is executed before any test is started.
        """
        delete_temp_dir(TEMP_DIR, recursive=True)
        if not os.path.isdir(TEMP_DIR):
            os.mkdir(TEMP_DIR)

        if os.path.isfile(META_FILE):
            os.remove(META_FILE)

    def setUp(self):
        """Prepare the testing environment."""
        self.config = open_config(CONF_FILE)
        self.train_file = os.path.join(TEMP_DIR, 'train_data.tsv')
        self.ann_file = os.path.join(TEMP_DIR, 'Cypripedium_section.ann')
        self.test_result = os.path.join(TEMP_DIR, 'test_result.tsv')
        self.train_dir = os.path.join(TEMP_DIR, 'train_data')
        self.ann_dir = os.path.join(TEMP_DIR, 'ann_dir')
        self.test_result_batch = os.path.join(TEMP_DIR, 'test_result_batch.tsv')

        for path in (self.train_dir, self.ann_dir):
            if not os.path.isdir(path):
                os.mkdir(path)

    def test_trainer_aa(self):
        """Test the `meta` subcommands."""
        sys.stdout.write("Initializing database...\n")
        db.make_meta_db(META_FILE)

        with db.session_scope(META_FILE) as (session, metadata):
            mkmeta = db.MakeMeta(self.config, IMAGE_DIR)
            mkmeta.make(session, metadata)

    def test_trainer_ab(self):
        """Test the `data` subcommands."""
        filter_ = self.config.classification.filter.as_dict()

        with db.session_scope(META_FILE) as (session, metadata):
            cache = PhenotypeCache()
            cache.make(IMAGE_DIR, TEMP_DIR, self.config, update=False)

            train_data = MakeTrainData(self.config, TEMP_DIR)
            train_data.export(self.train_file, filter_)

    def test_trainer_ac(self):
        """Test the `ann` subcommands."""
        ann_maker = MakeAnn(self.config)
        ann_maker.train(self.train_file, self.ann_file)

    def test_trainer_ad(self):
        """Test the `classify` subcommands."""
        filter_ = self.config.classification.filter.as_dict()
        image = os.path.join(IMAGE_DIR,
            "Cypripedium/Arietinum/plectrochilum/14990382409.jpg")

        with db.session_scope(META_FILE) as (session, metadata):
            classes = db.get_classes_from_filter(session, metadata, filter_)
            if not classes:
                raise ValueError("No classes found for filter `%s`" % filter_)
            codewords = get_codewords(classes)

            classifier = ImageClassifier(self.config)
            codeword = classifier.classify_image(image, self.ann_file,
                self.config)
            classification = get_classification(codewords, codeword, 0.001)

        class_ = [class_ for mse,class_ in classification]
        print "Image is classified as {0}".format(", ".join(class_))

    def test_trainer_ae(self):
        """Test the `test-ann` subcommands."""
        filter_ = self.config.classification.filter.as_dict()
        with db.session_scope(META_FILE) as (session, metadata):
            tester = TestAnn(self.config)
            tester.test(self.ann_file, self.train_file)
            tester.export_results(self.test_result, filter_, 0.001)

    def test_trainer_ba(self):
        """Test the `data-batch` subcommands."""
        with db.session_scope(META_FILE) as (session, metadata):
            cache = PhenotypeCache()
            cache.make(IMAGE_DIR, TEMP_DIR, self.config, update=False)

            train_data = BatchMakeTrainData(self.config, TEMP_DIR)
            train_data.batch_export(self.train_dir)

    def test_trainer_bb(self):
        """Test the `ann-batch` subcommands."""
        with db.session_scope(META_FILE) as (session, metadata):
            ann_maker = BatchMakeAnn(self.config)
            ann_maker.batch_train(self.train_dir, self.ann_dir)

    def test_trainer_bc(self):
        """Test the `test-ann-batch` subcommands."""
        with db.session_scope(META_FILE) as (session, metadata):
            tester = TestAnn(self.config)
            tester.test_with_hierarchy(self.train_dir, self.ann_dir, 0.001)

        correct, total = tester.export_hierarchy_results(self.test_result_batch)
        print "Correctly classified: {0}/{1} ({2:.2%})\n".\
            format(correct, total, float(correct)/total)

    def test_trainer_ca(self):
        """Test the `validate` subcommand.

        Should fail because not every class has enough photos.
        """
        self.assertRaisesRegexp(
            AssertionError,
            "The minimum number of labels for any class cannot be less than k",
            validate,
            self.config,
            k=4
        )

    def test_trainer_cb(self):
        """Test the `validate` subcommand.

        Should fail because there are no classes with at least 5 photos.

        .. note::

           Different scikit-learn versions raise different exception types.
        """
        self.assertRaisesRegexp(
            (AssertionError, ValueError),
            "Cannot have number of folds .* greater than the number of samples",
            validate,
            self.config,
            k=5,
            autoskip=True
        )

    def test_trainer_cc(self):
        """Test the `validate` subcommand.

        Should only process photos from classes with at least k=4 photos.
        """
        validate(self.config, k=4, autoskip=True)

    def test_trainer_cd(self):
        """Test the `validate` subcommand.

        Should be able to process all photos.
        """
        validate(self.config, k=3)

if __name__ == '__main__':
    unittest.main()
