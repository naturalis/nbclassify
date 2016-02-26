==========
NBClassify
==========

NBClassify is a collection of Python scripts for image phenotyping (using
the ImgPheno_ package) and recognition via artificial neural networks.

This package comes with the following scripts:

* ``nbc-harvest-images``: Download images with meta data from a Flickr
  account and save them in a directory hierarchy matching the the
  classifications.

* ``nbc-trainer``: Extract phenotypes from images, export these to
  training data, and train artificial neural networks.

* ``nbc-classify``: Identify user-submitted images using trained artificial
  neural networks.

* ``nbc-add-tags``: Add tags from a local spreadsheet file to images on Flickr.


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

* `Python Flickr API`_ (1.4.5)

* PyYAML_

* scikit-learn_ (>=0.15)

* SciPy

* SQLAlchemy_ (>=0.9.1)

On Debian (based) systems, most dependencies can be installed from the
software repository::

    apt-get install python-flickrapi python-numpy python-opencv python-pyfann \
    python-sklearn python-sqlalchemy python-yaml

More recent versions of some Python packages can be obtained via the Python
Package Index::

    pip install -r requirements.txt

The ImgPheno_ package can be installed from the GitHub repository.

Installation
============

From the GitHub repository::

    git clone https://github.com/naturalis/nbclassify.git
    pip install nbclassify/nbclassify/

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
