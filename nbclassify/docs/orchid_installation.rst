.. _orchid-setup:

====================
OrchID Installation
====================

The easiest and fastest method of deploying OrchID on a server is with Puppet
and the `Puppet OrchID module <https://github.com/naturalis/puppet-orchid>`_. If
instead you want to opt for the labour intensive method, continue with the rest
of this page.

Dependencies
------------

The OrchID Django app has the following dependencies:

* Django (>=1.7)
* Django REST framework
* FANN (>=2.1.0)

  * Python bindings

* ImgPheno_
* NBClassify_
* NumPy
* OpenCV (2.4.x)

  * Python bindings

* Python (>=2.7 && <2.8)

  * SQLite (>=3.6.19)

* PyYAML
* SciPy
* scikit-learn (>=0.15)
* sorl-thumbnail (>=12.2)

  * Pillow
  * Python-memcached
  * memcached

* SQLAlchemy (>=0.9.1)

On Debian (based) systems, most dependencies can be installed from the
software repository::

    apt-get install memcached python-django python-memcache python-numpy \
    python-opencv python-pil python-pyfann python-scipy python-sklearn \
    python-sorl-thumbnail python-sqlalchemy python-yaml

More recent versions of some Python packages can be obtained and built via the
Python Package Index::

    apt-get install python-pip python-dev gfortran libopenblas-dev liblapack-dev
    pip install -r requirements.txt

Follow the setup instructions for `sorl-thumbnail`_.

ImgPheno_ and NBClassify_ can be installed from the GitHub repositories.


Installation
------------

In a production environment it is recommended to install Python packages in a
virtual environment. But since system Python packages are also required (e.g.
python-opencv), the following could be done::

  cd /var/www/django-site/
  virtualenv --system-site-packages env
  env/bin/pip install -r requirements.txt


Quick start
-----------

1. Add "orchid" to your INSTALLED_APPS setting like this::

      INSTALLED_APPS = (
          ...
          'orchid',
      )

2. Include the orchid URLconf in your project urls.py like this::

      url(r'^orchid/', include('orchid.urls')),

3. Run ``python manage.py migrate`` to create OrchID's database tables.

4. You may start the development server with ``python manage.py runserver``
   and visit http://127.0.0.1:8000/ to test the app.


Deploying on Apache with mod_wsgi
---------------------------------

This setup assumes you have Apache 2.4.

1. Make sure that mod_wsgi is enabled::

      apt-get install libapache2-mod-wsgi
      a2enmod wsgi

2. Configure Apache and mod_wsgi for hosting a WSGI application (i.e. Django).
   The `mod_wsgi documentation`_ is a good place to start.

   A complete virtual host configuration for hosting this Django site in daemon
   mode could be something like::

      <VirtualHost *:80>
          ServerName example.com
          ServerAdmin webmaster@example.com

          WSGIDaemonProcess orchid deadlock-timeout=10 python-path=/var/www/orchid:/var/www/orchid/env/lib/python2.7/site-packages
          WSGIProcessGroup orchid
          WSGIApplicationGroup %{GLOBAL}
          WSGIScriptAlias / /var/www/orchid/webapp/wsgi.py

          Alias /media/ /var/www/orchid/media/
          Alias /static/admin/ /var/www/orchid/webapp/static/admin/
          Alias /static/ /var/www/orchid/orchid/static/

          <Directory /var/www/orchid/media>
              Require all granted
          </Directory>

          <Directory /var/www/orchid/webapp/static/admin>
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
   which points to a virtualenv directory. This is needed if Python packages
   were installed using virtualenv. For security reasons, a Django site (i.e.
   ``/var/www/orchid/``) must not be in the Apache document root. Notice that we
   made aliases for these paths:

   * ``/var/www/orchid/media/``
   * ``/var/www/orchid/webapp/static/...``
   * ``/var/www/orchid/orchid/static/``

   This way, Apache can still serve static and user uploaded files. Also make
   sure that ``/var/www/orchid/media/`` exists and is writable to Apache.
   Aliases were also created for ``static/admin/`` and
   ``static/rest_framework/``, which are needed for the admin panel and the JSON
   API. Both could be system links::

      mkdir /var/www/orchid/webapp/static/
      cd /var/www/orchid/webapp/static/
      ln -s ../../env/lib/python2.7/site-packages/django/contrib/admin/static/admin/
      ln -s ../../env/lib/python2.7/site-packages/rest_framework/static/rest_framework/

   The ``WSGIApplicationGroup`` directive is necessary because WingID depends on
   some Python modules that are affected by the `Simplified GIL State API`_
   issue.

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


.. _ImgPheno: https://github.com/naturalis/imgpheno
.. _NBClassify: https://github.com/naturalis/nbclassify
.. _`sorl-thumbnail`: http://sorl-thumbnail.readthedocs.org/en/latest/installation.html

.. _`mod_wsgi documentation`: https://code.google.com/p/modwsgi/wiki/QuickConfigurationGuide
.. _`Simplified GIL State API`: https://code.google.com/p/modwsgi/wiki/ApplicationIssues#Python_Simplified_GIL_State_API
