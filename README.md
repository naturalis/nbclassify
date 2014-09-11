# Image Classification

This repository contains code and examples that demonstrate the ability for
artificial neural networks to classify images of orchid species. The layout
is as follows:

* `doc/`: Contains documentation.
* `html/`: Contains a web application for image classification. It implements
  the `nbclassify` package.
* `nbclassify/`: A Python package for training artificial neural networks and
   image classification. It implements the [ImgPheno][1] package.


## Dependencies

* FANN (>=2.1.0)
  * Python bindings
* Numpy
* OpenCV (2.4.x)
  * Python bindings
* Python (>=2.7 && <2.8)
  * SQLite (>=3.6.19)
* Python Flickr API
* PyYAML
* [ImgPheno][1]
* SQLAlchemy

On Debian (based) systems, most dependencies can be installed from the
software repository:

    apt-get install libfann2 opencv python python-pyfann python-opencv \
    python-flickrapi python-numpy python-sqlalchemy python-yaml

More recent versions of some Python packages can be obtained via the Python
Package Index:

    pip install flickrapi numpy sqlalchemy pyyaml


[1]: https://github.com/naturalis/feature-extraction
