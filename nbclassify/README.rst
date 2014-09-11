==========
NBClassify
==========

NBClassify is a collection of Python scripts for extracting features from
images (using the ImgPheno_ package), generating training data from image
features, training artificial neural networks, and image classification.

This package comes with the following scripts:

* ``nbc-harvest-images.py``: Image harvester for downloading images with meta
  data from Flickr.

* ``nbc-trainer.py``: This script can be used to extract digital phenotypes
  from images, export these to training data files, and train and test
  artificial neural networks.

* ``nbc-classify.py``: Classify digital images using artificial neural
  networks.


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

.. _ImgPheno: https://github.com/naturalis/feature-extraction
.. _FANN: http://leenissen.dk/fann/wp/
.. _NumPy: http://www.numpy.org/
.. _OpenCV: http://opencv.org/
.. _Python: https://www.python.org/
.. _`Python Flickr API`: https://pypi.python.org/pypi/flickrapi
.. _PyYAML: https://pypi.python.org/pypi/PyYAML
.. _SQLAlchemy: http://www.sqlalchemy.org/
