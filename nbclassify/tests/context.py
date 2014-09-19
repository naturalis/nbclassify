# -*- coding: utf-8 -*-

import sys
import os.path
import importlib

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import nbclassify

nbc_trainer = importlib.import_module('scripts.nbc-trainer')
