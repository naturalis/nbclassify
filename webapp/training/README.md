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

# Training

The workflow for training the artificial neural networks (ANNs) is as follows:

1. `training.sh`

   a. `Offlickr.py`: Download photographs and metadata from Flickr.

   b. `get_tags.py`: Extract tags from the metadata.

   c. Manually check if all photos are downloaded correctly. If not,
      download the broken pictures by hand.

   d. Convert images to PNG format and move to correct directory.

2. `create_traindata.pl`

   a. `traindata.pl`: Create training data for tubers.

   b. `traindata2.pl`: Create training data for flowers.

3. `modify_flower_data.sh`

   a. `combine_files.py`: Create tab separated value (TSV) files per section.

   b. `add_columns.py`: Add classification columns to the TSV files.

4. `trainai.pl`

   a. Create and train ANNs for slipper orchid flower photographs or salep
      orchid tuber photographs.
