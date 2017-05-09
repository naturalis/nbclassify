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

* ``nbc-bag-of-words``: Pseudo-homologizes variable-length SURF features. 

* ``nbc-create-mean-bgr-file``: Creates BGR features files. Needs to be 
  refactored so that feature extraction is in imgpheno. Maybe the remaining
  program logic needs to be in nbc-trainer.

* ``nbc-feature-extraction``: Creates SURF features files. Needs to be 
  refactored so that feature extraction is in imgpheno. Maybe the remainined
  program logic needs to be in nbc-trainer.
