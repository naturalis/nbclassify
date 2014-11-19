# -*- coding: utf-8 -*-

import sys
import os.path
import importlib

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import nbclassify.base as base
from nbclassify.config import conf
import nbclassify.db as db
import nbclassify.functions as functions

# Disable FileExistsError exceptions.
conf.force_overwrite = True

# Raise exceptions which would otherwise be caught.
conf.debug = True

nbc_trainer = importlib.import_module('scripts.nbc-trainer')
