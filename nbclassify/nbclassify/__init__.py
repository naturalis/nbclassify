# -*- coding: utf-8 -*-

"""Image classification package for Python.

Provides scripts for training and classification. This package depends on the
`imgpheno <https://github.com/naturalis/imgpheno>`_ package for image feature
extraction.
"""

from .config import conf, ANN_DEFAULTS
from .functions import open_config

__version__ = '0.1.0'
