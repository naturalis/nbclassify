#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import tempfile
import unittest

from context import nbclassify as nbc
from context import nbc_trainer

CONF_FILE  = "config.yml"

# Disable FileExistsError exceptions.
nbc_trainer.FORCE_OVERWRITE = True


def delete_temp_file(path):
    """Delete temporary file."""
    if hasattr(f, 'file') and path.startswith(tempfile.gettempdir()):
        f.close()
        os.remove(path)

def delete_temp_dir(path, recursive=False):
    """Delete temporary directory with content."""
    if os.path.isdir(str(path)):
        if path.startswith(tempfile.gettempdir()):
            if recursive:
                shutil.rmtree(path)
            else:
                os.rmdir(path)

class TestTrainer(unittest.TestCase):

    """Unit tests for the trainer script."""

    def setUp(self):
        """Prepare the testing environment."""
        # Set the command-line arguments.
        self.argv_pre = ['exec', CONF_FILE]

        # Set the cache directory.
        self.temp_dir = os.path.join(tempfile.gettempdir(), 'nbclassify')

    #@unittest.skip("Debugging")
    def test_1(self):
        """Test the `{data|ann|classify|test-ann}` subcommands."""
        train_file = os.path.join(self.temp_dir, 'train_data.tsv')
        ann_file = os.path.join(self.temp_dir, 'Arietinum_species.ann')
        test_result = os.path.join(self.temp_dir, 'test_result.tsv')

        sys.argv = self.argv_pre + [
            'data',
            '--cache-dir', self.temp_dir,
            '-o', train_file,
            'images/'
        ]

        # Create an empty temporary directory.
        delete_temp_dir(self.temp_dir, recursive=True)
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)

        sys.stderr.write("\nRunning : {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'ann',
            '-o', ann_file,
            train_file
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'classify',
            '--ann', ann_file,
            '--imdir', 'images/',
            "images/Cypripedium/subgenus_null/Arietinum/plectrochilum/14990382409.jpg"
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'test-ann',
            '--ann', ann_file,
            '--error', '0.001',
            '-t', train_file,
            '-o', test_result,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()

        self.assertEqual(ret, 0)

    #@unittest.skip("Debugging")
    def test_2(self):
        """Test the `{data|ann|classify}-batch` subcommands."""
        train_dir = os.path.join(self.temp_dir, 'train_data')
        ann_dir = os.path.join(self.temp_dir, 'ann_dir')
        test_result = os.path.join(self.temp_dir, 'test_result_batch.tsv')

        for path in (train_dir, ann_dir):
            if not os.path.isdir(path):
                os.mkdir(path)

        sys.argv = self.argv_pre + [
            'data-batch',
            '--cache-dir', self.temp_dir,
            '-o', train_dir,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'ann-batch',
            '--data', train_dir,
            '-o', ann_dir,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'test-ann-batch',
            '--anns', ann_dir,
            '--test-data', train_dir,
            '-o', test_result,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()

        self.assertEqual(ret, 0)

    #@unittest.skip("Debugging")
    def test_3(self):
        """Test the `validate` subcommand."""
        sys.argv = self.argv_pre + [
            'validate',
            '--cache-dir', self.temp_dir,
            '-k4',
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()

        # Should fail because of not enough members per class.
        self.assertEqual(ret, 1)

        sys.argv = self.argv_pre + [
            'validate',
            '--cache-dir', self.temp_dir,
            '-k4',
            '--autoskip',
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()

        # Should fail because of no data.
        self.assertEqual(ret, 1)

        sys.argv = self.argv_pre + [
            'validate',
            '--cache-dir', self.temp_dir,
            '-k3',
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

if __name__ == '__main__':
    unittest.main()
