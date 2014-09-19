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


class TestTrainer(unittest.TestCase):

    """Unit tests for the trainer script."""

    def setUp(self):
        """Prepare the testing environment."""
        self.tmp = {}

        # Set the command-line arguments.
        self.argv_pre = ['exec', CONF_FILE]

        # Disable FileExistsError exceptions.
        nbc_trainer.FORCE_OVERWRITE = True

        # Create temporary files and folders.
        self.tmp['train_file'] = tempfile.NamedTemporaryFile(
            prefix='train_data_',
            delete=False
        )
        self.tmp['ann_file'] = tempfile.NamedTemporaryFile(
            prefix='Cypripedium_Arietinum_species_',
            delete=False
        )
        self.tmp['test_result'] = tempfile.NamedTemporaryFile(
            prefix='test_result_',
            delete=False
        )
        self.tmp['train_dir'] = tempfile.mkdtemp(
            prefix='train_data_'
        )
        self.tmp['ann_dir'] = tempfile.mkdtemp(
            prefix='ann_dir_'
        )

    def tearDown(self):
        """Delete temporary files and folders."""
        for f in self.tmp.values():
            if hasattr(f, 'file'):
                f.close()
                os.remove(f.name)
            elif os.path.isdir(str(f)):
                if f.startswith(tempfile.gettempdir()):
                    shutil.rmtree(f)

    def test_data_ann_classify(self):
        """Test the `{data|ann|classify|test-ann}` subcommands."""
        sys.argv = self.argv_pre + [
            'data',
            '-o', self.tmp['train_file'].name,
            'images/'
        ]

        sys.stderr.write("\nRunning : {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'ann',
            '-o', self.tmp['ann_file'].name,
            self.tmp['train_file'].name
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'classify',
            '--ann', self.tmp['ann_file'].name,
            '--imdir', 'images/',
            "images/Cypripedium/subgenus_null/Arietinum/plectrochilum/14990382409.jpg"
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'test-ann',
            '--ann', self.tmp['ann_file'].name,
            '--error', '0.001',
            '-t', self.tmp['train_file'].name,
            '-o', self.tmp['test_result'].name,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

    def test_batch(self):
        """Test the `{data|ann|classify}-batch` subcommands."""
        sys.argv = self.argv_pre + [
            'data-batch',
            '-o', self.tmp['train_dir'],
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'ann-batch',
            '--data', self.tmp['train_dir'],
            '-o', self.tmp['ann_dir'],
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

        sys.argv = self.argv_pre + [
            'test-ann-batch',
            '--anns', self.tmp['ann_dir'],
            '--test-data', self.tmp['train_dir'],
            '-o', self.tmp['test_result'].name,
            'images/'
        ]

        sys.stderr.write("\nRunning: {0}\n".format(' '.join(sys.argv)))
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)


if __name__ == '__main__':
    unittest.main()
