======
OrchiD
======

OrchiD is a proof-of-concept Django app for classifying digital images of
slipper orchids.

It implements the NBClassify Python package for image fingerprinting and
recognition. NBClassify depends on the `ImgPheno
<https://github.com/naturalis/imgpheno>`_ package and they both need to be
installed for this app to work.

Quick start
-----------

1. Add "orchid" to your INSTALLED_APPS setting like this::

      INSTALLED_APPS = (
          ...
          'orchid',
      )

2. Include the orchid URLconf in your project urls.py like this::

      url(r'^orchid/', include('orchid.urls')),

3. Run ``python manage.py syncdb`` to create OrchiD's database tables.

4. You may start the development server with ``python manage.py runserver``
   and visit http://127.0.0.1:8000/orchid/ to test the app.

   This app was however designed to be served on a production server (e.g. on
   Apache with mod_wsgi). Some functionality will not work using Django's
   development server. The following section explains how to deploy this app
   on Apache.

Deploying on Apache with mod_wsgi
---------------------------------

This setup assumes you have Apache 2.4.

1. Make sure that mod_wsgi is enabled::

      apt-get install libapache2-mod-wsgi
      a2enmod wsgi

2. Configure Apache and mod_wsgi for hosting a WSGI application (i.e. Django).
   The mod_wsgi documentation is a good place to start:
   https://code.google.com/p/modwsgi/wiki/QuickConfigurationGuide

   A complete virtual host configuration for hosting Django in daemon mode
   could be something like::

      <VirtualHost *:80>
          ServerName example.com
          ServerAdmin webmaster@example.com

          WSGIDaemonProcess orchid display-name=%{GROUP} python-path=/var/www/orchid:/var/www/orchid/env/lib/python2.7/site-packages
          WSGIProcessGroup orchid
          WSGIScriptAlias / /var/www/orchid/webapp/wsgi.py

          DocumentRoot /var/www/html

          Alias /media/ /var/www/orchid/media/
          Alias /static/ /var/www/orchid/orchid/static/

          <Directory /var/www/orchid/media>
              Require all granted
          </Directory>

          <Directory /var/www/orchid/orchid/static>
              Require all granted
          </Directory>

          <Directory /var/www/orchid/webapp>
              <Files wsgi.py>
                  Require all granted
              </Files>
          </Directory>
      </VirtualHost>

   In this example, we added the path
   ``/var/www/orchid/env/lib/python2.7/site-packages``
   which points to a virtualenv directory. This is needed if Python packaged
   were installed using virtualenv. For security reasons, a Django site (i.e.
   ``/var/www/orchid/``) must not be in the Apache document root. Notice that we
   made aliases for the paths ``/var/www/orchid/{orchid/static|media}/``. This
   way, Apache can still serve static and user uploaded files. Also make sure
   that ``/var/www/orchid/media/`` is writable to Apache.

   The corresponding ``settings.py`` for your Django site must have the
   following options set for this to work::

      STATIC_URL = '/static/'
      MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
      MEDIA_URL = '/media/'

   And if memcached is used for caching::

      CACHES = {
          'default': {
              'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache',
              'LOCATION': '127.0.0.1:11211',
          }
      }

   If you use an SQLite database, make sure that Apache can write to the
   database file and to the parent directory of the database.
