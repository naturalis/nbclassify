Installation
============

Requirements
------------

NBClassify has the following dependencies:

* FANN_ (>=2.1.0)

  * Python bindings

* ImgPheno_

* NumPy_

* OpenCV_ (2.4.x)

  * Python bindings

* Python_ (>=2.7 && <2.8)

  * SQLite (>=3.6.19)

* `Python Flickr API`_

* PyYAML_

* scikit-learn_ (>=0.15)

* SQLAlchemy_ (>=0.9.1)

On Debian (based) systems, most dependencies can be installed from the
software repository::

    apt-get install python-flickrapi python-numpy python-opencv python-pyfann \
    python-sklearn python-sqlalchemy python-yaml

More recent versions of some Python packages can be obtained via the Python
Package Index::

    pip install -r requirements.txt

The ImgPheno_ package can be installed from the GitHub repository::

    git clone https://github.com/naturalis/imgpheno.git
    cd imgpheno/
    python setup.py install

Installing NBClassify
---------------------

From the GitHub repository::

    git clone https://github.com/naturalis/nbclassify.git
    cd nbclassify/nbclassify/
    python setup.py install

Or if you have a source archive file::

    pip install nbclassify-0.1.0.tar.gz


.. _ImgPheno: https://github.com/naturalis/imgpheno
.. _FANN: http://leenissen.dk/fann/wp/
.. _NumPy: http://www.numpy.org/
.. _OpenCV: http://opencv.org/
.. _Python: https://www.python.org/
.. _`Python Flickr API`: https://pypi.python.org/pypi/flickrapi
.. _PyYAML: https://pypi.python.org/pypi/PyYAML
.. _scikit-learn: http://scikit-learn.org
.. _SQLAlchemy: http://www.sqlalchemy.org/
