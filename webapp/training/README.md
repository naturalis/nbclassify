# Requirements

## Offlickr.py

* libxml2 (with Python bindings)
* flickrapi

To install on Debian (based) systems:

    sudo apt-get install libxml2 python-libxml2 python-flickrapi

To install on Mac OS via pip and Homebrew:

    sudo easy_install pip
    sudo pip install flickrapi
    brew install --with-python libxml2

When the `brew` command fails:

1. Run `brew edit libxml2` and change "--without-python" into "--with-python"
2. Run the command again


## prepare_training.sh

* ImageMagick

To install on Debian (based) systems:

    sudo apt-get install imagemagick

To install on Mac OS via Homebrew:

    brew install imagemagick
