#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import tempfile
import unittest

from context import nbc
from context import nbc_trainer

CONF_FILE  = "config.yml"
META_FILE = ".meta.db"

# Temporary directory.
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'nbclassify-{0}'.format(os.getuid()))

# Disable FileExistsError exceptions.
nbc_trainer.FORCE_OVERWRITE = True

# Raise exceptions which would otherwise be caught.
nbc_trainer.DEBUG = True

def delete_temp_dir(path, recursive=False):
    """Delete temporary directory with content."""
    if os.path.isdir(str(path)):
        if path.startswith(tempfile.gettempdir()):
            if recursive:
                shutil.rmtree(path)
            else:
                os.rmdir(path)

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

        meta_file = os.path.join('images', META_FILE)
        if os.path.isfile(meta_file):
            os.remove(meta_file)

    def setUp(self):
        """Prepare the testing environment."""
        # Simulate running the scripts from the command-line.
        sys.argv = ['nbc-trainer.py', CONF_FILE]

        # Set paths.
        self.train_file = os.path.join(TEMP_DIR, 'train_data.tsv')
        self.ann_file = os.path.join(TEMP_DIR, 'Arietinum_species.ann')
        self.test_result = os.path.join(TEMP_DIR, 'test_result.tsv')
        self.train_dir = os.path.join(TEMP_DIR, 'train_data')
        self.ann_dir = os.path.join(TEMP_DIR, 'ann_dir')
        self.test_result_batch = os.path.join(TEMP_DIR, 'test_result_batch.tsv')

    def test_trainer_aa(self):
        """Test the `meta` subcommands."""
        sys.argv += [
            'meta',
            'images/'
        ]

        sys.stderr.write("\nRunning : {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_trainer_ab(self):
        """Test the `data` subcommands."""
        sys.argv += [
            'data',
            '--cache-dir', TEMP_DIR,
            '-o', self.train_file,
            'images/'
        ]

        sys.stderr.write("\nRunning : {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_trainer_ac(self):
        """Test the `ann` subcommands."""
        sys.argv += [
            'ann',
            '-o', self.ann_file,
            self.train_file
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_trainer_ad(self):
        """Test the `classify` subcommands."""
        sys.argv += [
            'classify',
            '--ann', self.ann_file,
            '--imdir', 'images/',
            "images/Cypripedium/Arietinum/plectrochilum/14990382409.jpg"
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_trainer_ae(self):
        """Test the `test-ann` subcommands."""
        sys.argv += [
            'test-ann',
            '--ann', self.ann_file,
            '--error', '0.001',
            '-t', self.train_file,
            '-o', self.test_result,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_trainer_ba(self):
        """Test the `data-batch` subcommands."""
        for path in (self.train_dir, self.ann_dir):
            if not os.path.isdir(path):
                os.mkdir(path)

        sys.argv += [
            'data-batch',
            '--cache-dir', TEMP_DIR,
            '-o', self.train_dir,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_trainer_bb(self):
        """Test the `ann-batch` subcommands."""
        sys.argv += [
            'ann-batch',
            '--data', self.train_dir,
            '-o', self.ann_dir,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_trainer_bc(self):
        """Test the `test-ann-batch` subcommands."""
        sys.argv += [
            'test-ann-batch',
            '--anns', self.ann_dir,
            '--test-data', self.train_dir,
            '-o', self.test_result_batch,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()

        self.assertEqual(ret, 0)

    def test_trainer_ca(self):
        """Test the `validate` subcommand.

        Should fail because not every class has enough photos.
        """
        sys.argv += [
            'validate',
            '--cache-dir', TEMP_DIR,
            '-k4',
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        self.assertRaisesRegexp(
            AssertionError,
            "The minimum number of labels for any class cannot be less than k",
            nbc_trainer.main
        )

    def test_trainer_cb(self):
        """Test the `validate` subcommand.

        Should fail because there are no classes with at least 5 photos.

        .. note::

           Different scikit-learn versions raise different exception types.
        """
        sys.argv += [
            'validate',
            '--cache-dir', TEMP_DIR,
            '-k5',
            '--autoskip',
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        self.assertRaisesRegexp(
            (AssertionError, ValueError),
            "Cannot have number of folds .* greater than the number of samples",
            nbc_trainer.main
        )

    def test_trainer_cc(self):
        """Test the `validate` subcommand.

        Should only process photos from classes with at least k=4 photos.
        """
        sys.argv += [
            'validate',
            '--cache-dir', TEMP_DIR,
            '-k4',
            '--autoskip',
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_trainer_cd(self):
        """Test the `validate` subcommand.

        Should be able to process all photos.
        """
        sys.argv += [
            'validate',
            '--cache-dir', TEMP_DIR,
            '-k3',
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

if __name__ == '__main__':
    unittest.main()
