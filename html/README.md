# OrchiD

## Dependencies

The OrchiD Django app has the following dependencies:

* Django (>=1.6)
* FANN (>=2.1.0)
  * Python bindings
* [ImgPheno][1]
* memcached
* NumPy
* OpenCV (2.4.x)
  * Python bindings
* Python (>=2.7 && <2.8)
  * SQLite (>=3.6.19)
* Python Imaging Library (PIL) or Pillow
* Python-memcached
* PyYAML
* sorl-thumbnail
* SQLAlchemy (>=0.9.1)

On Debian (based) systems, most dependencies can be installed from the
software repository:

    apt-get install memcached python-django python-memcache python-numpy \
    python-opencv python-pil python-pyfann python-sorl-thumbnail \
    python-sqlalchemy python-yaml


More recent versions of some Python packages can be obtained via the Python
Package Index:

    apt-get install python-pip

    pip install -r requirements.txt

Follow the setup instructions for [sorl-thumbnail][2].

The [ImgPheno][1] package can be installed from the GitHub repository.


## Installation

In a production environment it is recommended to install Python packages in a
virtual environment. But since system Python packages are also required (e.g.
python-opencv), the following could be done:

	cd /var/www/django-site/
	virtualenv --system-site-packages env
	env/bin/pip install -r requirements.txt

For further installation instructions, see the "orchid" subdirectory.


[1]: https://github.com/naturalis/imgpheno
[2]: http://sorl-thumbnail.readthedocs.org/en/latest/installation.html
