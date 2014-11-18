# -*- coding: utf-8 -*-

import sys
import os.path
import importlib

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import nbclassify.config as config
import nbclassify.base as base
import nbclassify.db as db
import nbclassify.functions as functions

# Disable FileExistsError exceptions.
config.FORCE_OVERWRITE = True

# Raise exceptions which would otherwise be caught.
config.DEBUG = True

nbc_trainer = importlib.import_module('scripts.nbc-trainer')
