# -*- coding: utf-8 -*-

import logging
import os
import tempfile

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(),
	'nbclassify-{0}'.format(os.getuid()))
CONF_FILE  = os.path.join(BASE_DIR, "config.yml")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# Display log messages.
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
