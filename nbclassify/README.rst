==========
NBClassify
==========

NBClassify is a collection of Python scripts for image fingerprinting (using
the ImgPheno_ package) and recognition via artificial neural networks.

This package comes with the following scripts:

* ``nbc-harvest-images.py``: Download images with meta data from a Flickr
  account. This also builds an SQLite database with image meta data for use
  in downsteam scripts ``nbc-trainer.py`` and ``nbc-classify.py``.

* ``nbc-trainer.py``: Extract fingerprints from images, export these to
  training data, and train artificial neural networks.

* ``nbc-classify.py``: Recognize images using artificial neural networks.


Requirements
============

This Python package has the following dependencies:

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

* SQLAlchemy_

On Debian (based) systems, most dependencies can be installed from the
software repository::

    apt-get install libfann2 opencv python python-pyfann python-opencv \
    python-flickrapi python-numpy python-sqlalchemy python-yaml

More recent versions of some Python packages can be obtained via the Python
Package Index::

    pip install flickrapi numpy sqlalchemy pyyaml

The ImgPheno_ package can be installed from the GitHub repository.

.. _ImgPheno: https://github.com/naturalis/feature-extraction
.. _FANN: http://leenissen.dk/fann/wp/
.. _NumPy: http://www.numpy.org/
.. _OpenCV: http://opencv.org/
.. _Python: https://www.python.org/
.. _`Python Flickr API`: https://pypi.python.org/pypi/flickrapi
.. _PyYAML: https://pypi.python.org/pypi/PyYAML
.. _SQLAlchemy: http://www.sqlalchemy.org/
