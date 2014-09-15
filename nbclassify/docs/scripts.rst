.. highlight:: console

=================
Using the Scripts
=================

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

The workflow for the scripts is as follows:

.. graphviz::

   digraph scripts {
        node [shape=ellipse]; classification;
        node [shape=parallelogram]; Flickr; "new image";
        node [shape=box,style=filled]; "nbc-harvest-images.py"; "nbc-trainer.py"; "nbc-classify.py";

        Flickr -> "nbc-harvest-images.py";
        "new image" -> "nbc-classify.py";
        "nbc-harvest-images.py" -> "nbc-trainer.py" [ label=" images\n meta data" ];
        "nbc-trainer.py" -> "nbc-classify.py" [ label=" neural networks" ];
        "nbc-classify.py" -> "classification";
   }

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

See the ``--help`` option for any of these subcommands for usage information.

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

Downloaded images are placed in a directory hierarchy:

.. code-block:: text

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

    $ nbc-harvest-images.py -v 123456789@A12 harvest -o images/orchids/ \
    > --page 1 --per-page 500


.. _nbc-trainer-py:

nbc-trainer.py
==============

Used to extract fingerprints, or "phenotypes", from digital images, export
these to training data files, and train and test artificial neural networks.

This script uses a configurations file which controls how images are processed
and how neural networks are trained. See :ref:`config` for detailed
information.

This script depends on the SQLite database file with meta data for a Flickr
harvested image collection. This database is created by
:ref:`nbc-harvest-images-py`, which is also responsible for archiving the
images in a local directory.

The following subcommands are available:

:ref:`nbc-trainer-py-data`
  Create a tab separated file with training data.

:ref:`nbc-trainer-py-batch-data`
  Batch create tab separated files with training data.

:ref:`nbc-trainer-py-ann`
  Train an artificial neural network.

:ref:`nbc-trainer-py-batch-ann`
  Batch train artificial neural networks.

:ref:`nbc-trainer-py-test-ann`
  Test an artificial neural network.

:ref:`nbc-trainer-py-test-ann-batch`
  Test the artificial neural networks for a classification hierarchy.

:ref:`nbc-trainer-py-classify`
  Classify an image using a single neural network.

See the ``--help`` option for any of these subcommands for usage information.


-----------
Subcommands
-----------

.. _nbc-trainer-py-data:

data
----

Create a tab separated file with training data. :ref:`Preprocessing steps
<config-preprocess>`, :ref:`features to extract <config-features>`, and a
:ref:`classification filter <config-classification.filter>` must be set in a
configurations file.

Example usage::

    $ nbc-trainer.py data --conf config.yml -o train_data.tsv images/orchids/


.. _nbc-trainer-py-batch-data:

batch-data
----------

In contrast to the :ref:`nbc-trainer-py-data` subcommand, this will
automatically create all the training data files needed to train neural
networks for classification on each level in a :ref:`classification hierarchy
<config-classification.hierarchy>`. It uses the classification hierarchy to
determine which training data files need to be created.

Example usage::

    $ nbc-trainer.py batch-data --conf config.yml -o train_data/ images/orchids/


.. _nbc-trainer-py-ann:

ann
----

Train an artificial neural network. Optional training parameters
:ref:`config-ann` can be set in a configurations file.

Example usage::

    $ nbc-trainer.py ann --conf config.yml -o orchid.ann train_data.tsv


.. _nbc-trainer-py-batch-ann:

batch-ann
---------

The batch equivalent of the :ref:`nbc-trainer-py-ann` subcommand, and similar
to the :ref:`nbc-trainer-py-batch-data` subcommand, in that it automatically
creates all the required artificial neural networks needed for classifying an
image on the levels specified in the :ref:`classification hierarchy
<config-classification.hierarchy>`. Training data required for this subcommand
is created with the :ref:`nbc-trainer-py-batch-data` subcommand.

Example usage::

    $ nbc-trainer.py batch-ann --conf config.yml \
    >  --db images/orchids/photos.db --data train_data/ -o anns/


.. _nbc-trainer-py-test-ann:

test-ann
---------

Test an artificial neural network. If ``--output`` is used, then ``--db`` must
be set and the :ref:`classification filter <config-classification.filter>`
must be set in the configurations file.

.. note::

   Test data has the same format as training data, except that the samples
   should contain data that is new to the neural network.

Example usage::

    $ nbc-trainer.py test-ann --conf config.yml --ann orchid.ann \
    > --db images/orchids/photos.db -o test-results.tsv --error 0.001 \
    > test_data.tsv


.. _nbc-trainer-py-test-ann-batch:

test-ann-batch
--------------

Test the artificial neural networks for a :ref:`classification hierarchy
<config-classification.hierarchy>`.

.. note::

   Use the :ref:`nbc-trainer-py-batch-data` subcommand with out-of-sample
   images to create a directory with test data for a classification
   hierarchy.

Example usage::

    $ nbc-trainer.py test-ann-batch --conf config.yml \
    > --db images/orchids/photos.db --test-data test_data/ \
    > --anns neural_networks/ --error 0.001 -o test-results.tsv


.. _nbc-trainer-py-classify:

classify
--------

Classify an image using a single neural network. The :ref:`classification
filter <config-classification.filter>` must be set in the configurations file.

Example usage::

    $ nbc-trainer.py classify --conf config.yml --ann orchid.ann \
    > --db images/orchids/photos.db --error 0.001 images/test/14371998807.jpg


.. _nbc-classify-py:

nbc-classify.py
===============

Classify digital images using artificial neural networks. Each image is
classified on different levels in a :ref:`classification hierarchy
<config-classification.hierarchy>`, which in this case is a taxonomic
hierarchy.

The neural networks on which this script depends are created with a separate
script, :ref:`nbc-trainer-py`. See its :ref:`nbc-trainer-py-batch-data` and
:ref:`nbc-trainer-py-batch-ann` subcommands for more information.

This script depends on the SQLite database file with meta data for a Flickr
harvested image collection. This database is created by
:ref:`nbc-harvest-images-py`, which is also responsible for archiving the
images in a local directory.

See the ``--help`` option for usage information.

Example usage::

    $ nbc-classify.py -v --conf config.yml --db images/orchids/photos.db \
    > --anns neural_networks/ images/test/14371998807.jpg
    Image: images/test/14371998807.jpg
    INFO Segmenting...
    INFO Extracting features...
    INFO - Running color:bgr_means...
    INFO Using ANN `neural_networks/genus.ann`
    INFO Level `genus` at node `/` classified as `Phragmipedium`
    INFO Using ANN `neural_networks/Phragmipedium.section.ann`
    INFO Branching in level `section` at node '/Phragmipedium' into `Micropetalum, Platypetalum`
    INFO Using ANN `neural_networks/Phragmipedium.Micropetalum.species.ann`
    INFO Level `species` at node `/Phragmipedium/Micropetalum` classified as `fischeri`
    INFO Using ANN `neural_networks/Phragmipedium.Platypetalum.species.ann`
    INFO Level `species` at node `/Phragmipedium/Platypetalum` classified as `sargentianum`
      Classification:
        genus: Phragmipedium
          section: Micropetalum
            species: fischeri
        Mean square error: 2.14122181117e-10
      Classification:
        genus: Phragmipedium
          section: Platypetalum
            species: sargentianum
        Mean square error: 0.000153084416316


.. _config.yml: https://github.com/naturalis/img-classify/blob/master/nbclassify/nbclassify/config.yml
