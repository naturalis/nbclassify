==================
NBClassify Scripts
==================

NBClassify comes with several command-line scripts. The following scripts are
included:

:ref:`nbc-harvest-images-py`
  Downloads images with meta data from a Flickr
  account. This also builds an SQLite database with image meta data for use
  in downstream scripts.

:ref:`nbc-trainer-py`
  Extract fingerprints from images, export these to
  training data, and train artificial neural networks.

:ref:`nbc-classify-py`
  Classify images using artificial neural networks.

The workflow for the scripts is as follows::

    .-------------------------.
    |  nbc-harvest-images.py  | <-- Flickr
    '-------------------------'
                |
             images
            meta data
                |
                v
       .------------------.
       |  nbc-trainer.py  |
       '------------------'
                |
         (training data)
         neural networks
                |
                v
      .-------------------.
      |  nbc-classify.py  | <-- new image
      '-------------------'
                |
                v
          classification

Each script is explained in more detail below.


.. _nbc-harvest-images-py:

nbc-harvest-images.py
=====================

Image harvester for downloading photos with meta data from Flickr.

The following subcommands are available:

:ref:`nbc-harvest-images-py-harvest`
  Download images with meta data from a Flickr account.

cleanup
  Clean up your local archive of Flickr harvested images. Images that were
  harvested, but were later removed from Flickr, will also be deleted from
  your local archive.

See the ``--help`` option for any of these subcommands for more information.

-----------
Subcommands
-----------

.. _nbc-harvest-images-py-harvest:

harvest
-------

Download images with meta data from a Flickr account. In Flickr, you can give
your images tags, and this script expects to find certain tags for the images
it downloads. These Flickr tags are used to specify the classification for
each image. The script understands the following tags:

* ``genus:*``

* ``section:*``

* ``species:*``

where ``*`` is the corresponding taxonomic rank for that image (e.g.
``genus:Phragmipedium`` ``section:Micropetalum`` ``species:besseae``). The
``section:*`` tag is not required, as not all slipper orchid species have been
subdivided into a section. The script will not download images for which the
required tags are not set.

Downloaded images are placed in a directory hierarchy::

    output_dir
        ├── photos.db
        └── <genus>
            └── <subgenus>
                └── <section>
                    └── <species>
                        ├── 123456789.jpg
                        └── ...

where ``output_dir`` is specified with the ``--output`` option, and ``<*>`` is
replaced by the corresponding taxonomic ranks found in the image tags. If a
specific taxonomic rank is not set in the tag, then the directory name for
that rank will be ``<rank_name>_null``. Image files are saved with their
Flickr ID as the filename (e.g. ``123456789.jpg``).

As the script downloads the images from Flickr, it will also save image meta
data to an SQLite database file, by default a file named ``photos.db`` in the
output directory. The SQL schema for this database can be found here:
:download:`photos.sql <../databases/photos.sql>`. This database file is used
by the downstream scripts (:ref:`nbc-trainer-py` and :ref:`nbc-classify-py`)
to locate the Flickr harvested images and their corresponding classifications.

Example usage::

    nbc-harvest-images.py -v 123456789@A12 harvest -o images/orchids/ --page 1 --per-page 500


.. _nbc-trainer-py:

nbc-trainer.py
==============

Used to extract fingerprints, or "phenotypes", from digital images, export
these to training data files, and train and test artificial neural networks.

This script uses a configurations file in the YAML_ format. This
configurations file specifies what preprocessing needs to be done on images,
what features need to be extracted from the images, what the classification
hierarchy looks like, and how the neural networks are trained. See
:download:`nbclassify/config.yml <../nbclassify/config.yml>` for an example
configurations file.

This script depends on the SQLite database file with meta data for a Flickr
harvested image collection. This database is created by
:ref:`nbc-harvest-images-py`, which is also responsible for archiving the
images in a local directory.

The following subcommands are available:

data
  Create a tab separated file with training data. Preprocessing steps,
  features to extract, and a classification filter must be set in a
  configurations file. See :download:`nbclassify/config.yml
  <../nbclassify/config.yml>` for an example.

:ref:`nbc-trainer-py-batch-data`
  Batch create tab separated files with training data. Preprocessing steps,
  features to extract, and the classification hierarchy must be set in a
  configurations file, See :download:`nbclassify/config.yml
  <../nbclassify/config.yml>` for an example.

ann
  Train an artificial neural network. Optional training parameters ``ann`` can
  be set in a separate configurations file. See
  :download:`nbclassify/config.yml <../nbclassify/config.yml>` for an example.

batch-ann
  Batch train a committee of artificial neural networks. The classification
  hierarchy with optionally neural network training parameters for each level
  must be set in a configurations file. See :download:`nbclassify/config.yml
  <../nbclassify/config.yml>` for an example.

test-ann
  Test an artificial neural network. If ``--output`` is used, then ``--db``
  must be set, and the classification filter must be set in the configurations
  file. See :download:`nbclassify/config.yml <../nbclassify/config.yml>` for
  an example.

test-ann-batch
  Test the artificial neural networks for a classification hierarchy. See
  :download:`nbclassify/config.yml <../nbclassify/config.yml>` for an example.

classify
  Classify a digital photo. The classification filter must be set in the
  configurations file. See :download:`nbclassify/config.yml
  <../nbclassify/config.yml>` for an example.

See the ``--help`` option for any of these subcommands for more information.


-----------
Subcommands
-----------

.. _nbc-trainer-py-batch-data:

batch-data
----------

In contrast to the ``data`` subcommand, this will automatically create all the
training data files needed to train neural networks for classification on each
level in the classification hierarchy. It uses the classification hierarchy
set in the configurations file to determine which training data files need to
be created.

A classification hierarchy consists of `levels`, and each level has a name. In
the case of the slipper orchids, the levels correspond to the taxanomic ranks:
genus, section, species. In this case, this means that each image can be
classified on three levels. This subcommand creates all the training data
files needed to train neural networks for classification on each level. This
is what a classification hierarchy looks like:

.. code-block:: yaml

    classification:
        hierarchy:
            - name: genus
              preprocess: *preprocess_std
              features: *features_std
              ann: *ann_genus
              train_file: genus.tsv
              test_file: genus.tsv
              ann_file: genus.ann
              max_error: 0.00001

            - name: section
              preprocess: *preprocess_std
              features: *features_std
              ann: *ann_std
              train_file: __genus__.section.tsv
              test_file: __genus__.section.tsv
              ann_file: __genus__.section.ann
              max_error: 0.0001

            - name: species
              preprocess: *preprocess_std
              features: *features_std
              ann: *ann_std
              train_file: __genus__.__section__.species.tsv
              test_file: __genus__.__section__.species.tsv
              ann_file: __genus__.__section__.species.ann
              max_error: 0.001

The order of the levels is important. In this example, each image is first
classified on the genus level. Once the genus is known, it will classify on
sections within that specific genus. And when the section is know, it will
classify on species within that section. As you can see, the
``{train|test|ann}_file`` directives support ``__level__`` wildcards, where
``level`` is replaced by the corresponding classification in the parent level.

The classification hierarcy supports the following options:

name*
  The name of the level.

preprocess
  Image preprocessing options.

features*
  The features that need to be extracted from the images.

ann*
  Settings for training the artificial neural networks.

train_file*
  File name for training data files.

test_file
  File name for test data files.

ann_file*
  File name for neural network files.

max_error
  Mean square error threshold for classification on this level. A
  classification is only accepted if the mean square error is below this
  value. A default error threshold can be set with the ``--error`` option.

Options marked with an asterisk (*) are required.

Example usage::

    nbc-trainer.py batch-data --conf config.yml -o train_data/ images/orchids/


.. _nbc-classify-py:

nbc-classify.py
===============

TODO


.. _YAML: http://yaml.org/
