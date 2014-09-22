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


def delete_temp_file(f):
    """Delete temporary file."""
    if hasattr(f, 'file') and f.name.startswith(tempfile.gettempdir()):
        f.close()
        os.remove(f.name)

def delete_temp_dir(path):
    """Delete temporary directory with content."""
    if os.path.isdir(str(path)):
        if path.startswith(tempfile.gettempdir()):
            shutil.rmtree(path)


class TestTrainer(unittest.TestCase):

    """Unit tests for the trainer script."""

    def setUp(self):
        """Prepare the testing environment."""
        # Set the command-line arguments.
        self.argv_pre = ['exec', CONF_FILE]

    #@unittest.skip("Debugging")
    def test_simple(self):
        """Test the `{data|ann|classify|test-ann}` subcommands."""

        train_file = tempfile.NamedTemporaryFile(
            prefix='train_data_',
            delete=False
        )
        ann_file = tempfile.NamedTemporaryFile(
            prefix='Cypripedium_Arietinum_species_',
            delete=False
        )
        test_result = tempfile.NamedTemporaryFile(
            prefix='test_result_',
            delete=False
        )

        sys.argv = self.argv_pre + [
            'data',
            '-o', train_file.name,
            'images/'
        ]

        sys.stderr.write("\nRunning : {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'ann',
            '-o', ann_file.name,
            train_file.name
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'classify',
            '--ann', ann_file.name,
            '--imdir', 'images/',
            "images/Cypripedium/subgenus_null/Arietinum/plectrochilum/14990382409.jpg"
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'test-ann',
            '--ann', ann_file.name,
            '--error', '0.001',
            '-t', train_file.name,
            '-o', test_result.name,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()

        # Delete temporary files.
        delete_temp_file(train_file)
        delete_temp_file(ann_file)
        delete_temp_file(test_result)

        self.assertEqual(ret, 0)

    #@unittest.skip("Debugging")
    def test_batch(self):
        """Test the `{data|ann|classify}-batch` subcommands."""

        train_dir = tempfile.mkdtemp(
            prefix='train_data_'
        )
        ann_dir = tempfile.mkdtemp(
            prefix='ann_dir_'
        )
        test_result = tempfile.NamedTemporaryFile(
            prefix='test_result_',
            delete=False
        )

        sys.argv = self.argv_pre + [
            'data-batch',
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
            '-o', test_result.name,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()

        # Delete temporary files and folders.
        delete_temp_dir(train_dir)
        delete_temp_dir(ann_dir)
        delete_temp_file(test_result)

        self.assertEqual(ret, 0)


if __name__ == '__main__':
    unittest.main()
