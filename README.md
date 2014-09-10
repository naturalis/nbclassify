# img-classify

This repository contains code and examples that demonstrate the ability for
artificial neural networks to classify images of plant species. The layout
is as follows:

* `data/databases/`: Contains schemas of databases used by any of the scripts.
* `doc/`: Contains documentation.
* `html/`: Contains a web application for image classification.
* `script/`: Contains the scripts.
* `script/docs/`: Contains the API documentation for the `nbclassify` package.
  You need [Sphinx][1] to build the documentation. Run `pip install -r
  requirements.txt` followed by `make html` to build the documentation.
* `script/nbclassify/`: A Python package with common code used by the scripts.
* `script/harvest-images.py`: Image harvester for downloading photos with meta
  data from Flickr.
* `script/trainer.py`: This script can be used to extract digital phenotypes
  from photos, export these to training data files, and train and test
  artificial neural networks.
* `script/classify.py`: Classify digital photos using artificial neural
  networks.
* `script/trainer.yml`: An example configuration file as used by `trainer.py`
  and `classify.py`.

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
* Python package for [image feature extraction][2]
* SQLAlchemy

On Debian (based) systems, most dependencies can be installed from the
software repository:

    sudo apt-get install libfann2 opencv python python-pyfann python-opencv \
    python-flickrapi python-numpy python-sqlalchemy python-yaml

More recent versions of some Python packages can be obtained via the Python
Package Index:

    sudo apt-get install python-pip

    sudo pip install -r requirements.txt

[1]: http://sphinx-doc.org/
[2]: https://github.com/naturalis/feature-extraction "Python package for image feature extraction"
