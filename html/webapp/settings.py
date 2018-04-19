"""
Django settings for webapp project.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.6/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'omybzuduyg2f!4s4=yuy$sk#&=+@$3-u5786u=@c@7)(sdw2)e'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

TEMPLATE_DEBUG = False

ALLOWED_HOSTS = ['plakvallen.naturalis.nl']


# Application definition

INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'geoposition',
    'sorl.thumbnail',
    'rest_framework',
    'sticky_traps',
)

MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
)

ROOT_URLCONF = 'webapp.urls'

WSGI_APPLICATION = 'webapp.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.6/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# Internationalization
# https://docs.djangoproject.com/en/1.6/topics/i18n/

LANGUAGE_CODE = 'en-gb'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# URL to use when referring to static files located in STATIC_ROOT.
# Example: "/static/" or "http://static.example.com/"
STATIC_URL = '/static/'

# Set the full path to the static root here, to avoid confusion over the
# templates that need to be used for the specific project.
# STATIC_ROOT = "full/path/to/sticky_traps/static/"

# The list of finder backends that know how to find static files in various
# locations.
STATICFILES_FINDERS = (
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder"
)

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/var/www/example.com/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, 'uploads')
# Set the full path to the uploads directory below if the media root cannot be
# found.
# MEDIA_ROOT = 'full/path/to/sticky_traps/uploads/'

# If the automated template loader has trouble finding the templates, use 
# TEMPLATES as shown below.
# The exact path was given for the sticky_traps project.
# TEMPLATES = [
    # {
        # "BACKEND": "django.template.backends.django.DjangoTemplates",
        # "DIRS": [
            # "full/path/to/sticky_traps/templates/sticky_traps",
            # "full/path/to/sticky_traps/templates",
        # ],
        # "OPTIONS": {
            # "context_processors": [
                # "django.contrib.auth.context_processors.auth",
            # ]
        # }
    # },
# ]

# URL that handles the media served from MEDIA_ROOT, used for managing stored
# files. It must end in a slash if set to a non-empty value. You will need to
# configure these files to be served in both development and production.
# Example: "http://media.example.com/"
MEDIA_URL = "set/full/path/to/uploads/"


# A dictionary containing the settings for all caches to be used with Django.
# https://docs.djangoproject.com/en/1.6/ref/settings/#caches
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache',
        'LOCATION': '127.0.0.1:11211', # Set server IP here.
    }
}

# Set permissions for the REST framework.
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny'
    ],
    'PAGINATE_BY': 30
}

DATE_INPUT_FORMATS = ('%d-%m-%Y')

GEOPOSITION_GOOGLE_MAPS_API_KEY = 'insert your own key here' # Set the API-key.

GEOPOSITION_MAP_OPTIONS = {
    'minZoom': 8,
    'maxZoom': 0,
}


GEOPOSITION_MAP_WIDGET_HEIGHT = 700
