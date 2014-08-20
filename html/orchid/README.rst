=================
NBClassify Orchid
=================

Orchid is a proof-of-concept Django app for classifying digital images of
orchids. It is part of the NBClassify Python package for automated
classification of images using artificial neural networks.

Quick start
-----------

1. Add "orchid" to your INSTALLED_APPS setting like this::

      INSTALLED_APPS = (
          ...
          'orchid',
      )

2. Include the orchid URLconf in your project urls.py like this::

      url(r'^orchid/', include('orchid.urls')),

3. Run ``python manage.py syncdb`` to create the orchid models.

4. You may start the development server with ``python manage.py runserver``
   and visit http://127.0.0.1:8000/orchid/ to test the app.

   This app was however designed to be served on a production server (e.g. on
   Apache with mod_wsgi). Some functionality will not work using Django's
   development server. The following section explains how to deploy this app
   on Apache.

Deploying on Apache with mod_wsgi
---------------------------------

This setup assumes you have Apache 2.4.

1. Make sure that `mod_wsgi` is enabled::

      sudo apt-get install libapache2-mod-wsgi
      sudo a2enmod wsgi

2. Configure Apache and mod_wsgi for hosting a WSGI application (i.e. Django).
   The mod_wsgi documentation is a good place to start:
   https://code.google.com/p/modwsgi/wiki/QuickConfigurationGuide

   A complete virtual host configuration for hosting Django in daemon mode
   could be something like::

      <VirtualHost *:80>
          ServerName mysite.com
          ServerAdmin webmaster@mysite.com

          DocumentRoot /usr/local/www/documents

          Alias /media/ /usr/local/www/documents/media/
          Alias /static/ /usr/local/www/documents/static/

          <Directory /usr/local/www/documents/>
              Options FollowSymLinks
              AllowOverride None
              Require all granted
          </Directory>

          WSGIDaemonProcess mysite.com display-name=%{GROUP} python-path=/path/to/mysite.com
          WSGIProcessGroup mysite.com

          WSGIScriptAlias / /path/to/mysite.com/webapp/wsgi.py

          <Directory /path/to/mysite.com/webapp>
              <Files wsgi.py>
                  Require all granted
              </Files>
          </Directory>
      </VirtualHost>

   For security reasons, a Django site (i.e. ``/path/to/mysite.com``) must not
   be in the Apache document root. In this example setup, the paths
   ``/usr/local/www/documents/{static|media}/`` could be system links to
   ``/path/to/mysite.com/{orchid/static|media}/``. This way, Apache can still
   serve static and user uploaded files. Also make sure that
   ``/path/to/mysite.com/media/`` is writable to Apache.

   If you use an SQLite database, make sure that Apache can write to the
   database file and to the parent directory of the database.

   The corresponding ``settings.py`` for your Django site must have the
   following options set for this to work::

      STATIC_URL = '/static/'
      MEDIA_ROOT = '/path/to/mysite.com/media/'
      MEDIA_URL = '/media/'
