"""
WSGI config for webapp project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.6/howto/deployment/wsgi/
"""

import os
import sys

# Set direct path to Django root directory
# path = 'set/path/to/html'
# if path not in sys.path:
    # sys.path.append(path)

os.environ["DJANGO_SETTINGS_MODULE"] = "webapp.settings"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
