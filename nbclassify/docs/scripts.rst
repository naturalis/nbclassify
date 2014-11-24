.. highlight:: console

=================
Using the Scripts
=================

NBClassify comes with several command-line scripts. The following scripts are
included:

:ref:`nbc-harvest-images`
  Downloads images with metadata from a Flickr account. This also compiles a
  metadata file in the image directory, which is needed for downstream scripts.

:ref:`nbc-trainer`
  Extract fingerprints from images, export these to training data, and train
  artificial neural networks.

:ref:`nbc-classify`
  Classify images using artificial neural networks.

The workflow for the scripts is as follows:

.. graphviz::

   digraph scripts {
        bgcolor="transparent";
        node [shape=ellipse]; classification;
        node [shape=parallelogram]; "new image";
        node [shape=box,style=filled]; "nbc-harvest-images"; "nbc-trainer"; "nbc-classify";

        "new image" -> "nbc-classify";
        "nbc-harvest-images" -> "nbc-trainer" [ label=" images\n metadata" ];
        "nbc-trainer" -> "nbc-classify" [ label=" neural networks" ];
        "nbc-trainer" -> "nbc-trainer" [ label=" training data" ];
        "nbc-classify" -> "classification";
   }

Each script is explained in more detail below.


.. _nbc-harvest-images:

nbc-harvest-images
==================

Image harvester for downloading photos with metadata from Flickr.

The following subcommands are available:

:ref:`nbc-harvest-images-harvest`
  Download images with metadata from a Flickr account.

cleanup
  Clean up your local archive of Flickr harvested images. Images that were
  harvested, but were later removed from Flickr, will also be deleted from
  your local archive.

See the ``--help`` option for any of these subcommands for usage information.

-----------
Subcommands
-----------

.. _nbc-harvest-images-harvest:

harvest
-------

Download images with metadata from a Flickr account. In Flickr, you can give
your images tags, and this script expects to find certain tags for the images it
downloads. These Flickr tags are used to specify the classification for each
image. The script understands the format ``rank:taxon`` for tags (e.g.
``genus:Phragmipedium`` ``section:Micropetalum`` ``species:besseae``). The tags
for genus and species are mandatory. The script will not download images for
which the genus or species tags are not set.

Downloaded images are placed in a directory hierarchy:

.. code-block:: text

    PATH
    ├── .meta.db
    └── <genus>
        └── <section>
            └── <species>
                ├── 123456789.jpg
                └── ...

where ``PATH`` is the target directory and ``<...>`` is replaced by the
corresponding taxonomic ranks found in the image tags. If a specific taxonomic
rank is not set in the tag, then the directory name for that rank will be
"None". Image files are saved with their Flickr ID as the filename (e.g.
``123456789.jpg``).

As the script downloads the images from Flickr, it will also save image metadata
to a file, by default a file named ``.meta.db`` in the target directory. This
database file is used by the downstream scripts (:ref:`nbc-trainer` and
:ref:`nbc-classify`) to locate the Flickr harvested images and their
corresponding classifications.

Example usage::

    $ nbc-harvest-images -v 123456789@A12 harvest \
    > --page 1 --per-page 500 images/orchids/


.. _nbc-trainer:

nbc-trainer
===========

Used to extract fingerprints, or "phenotypes", from digital images, export
these to training data files, and train and test artificial neural networks.

This script uses a configurations file which controls how images are processed
and how neural networks are trained. See :ref:`config` for detailed information.

Before this script can work with an image collection, a metadata file must first
be compiled for an image collection. This metadata file contains taxon
information for images in a directory. This file is automatically created by
:ref:`nbc-harvest-images` during harvesting of image, or can be manually
compiled for an existing image directory with the `meta` subcommand.

The following subcommands are available:

:ref:`nbc-trainer-meta`
  Compile a metadata file for a directory of images.

:ref:`nbc-trainer-data`
  Create a tab separated file with training data.

:ref:`nbc-trainer-data-batch`
  Create tab separated files with training data for a classification hierarchy.

:ref:`nbc-trainer-ann`
  Train an artificial neural network.

:ref:`nbc-trainer-ann-batch`
  Train artificial neural networks for a classification hierarchy.

:ref:`nbc-trainer-test-ann`
  Test an artificial neural network.

:ref:`nbc-trainer-test-ann-batch`
  Test the artificial neural networks for a classification hierarchy.

:ref:`nbc-trainer-classify`
  Classify an image using a single neural network.

:ref:`nbc-trainer-validate`
  Test the performance of trained neural networks. Performs stratified K-fold
  cross validation.

:ref:`nbc-trainer-taxa`
  Print the taxon hierarcy for the metadata of an image collection.

See the ``--help`` option for any of these subcommands for usage information.


-----------
Subcommands
-----------

.. _nbc-trainer-meta:

meta
----

Compile a metadata file for a directory of images. Images must be stored in a
:ref:`directory hierarchy <config-directory_hierarchy>`, which is described in
the configurations file. The metadata file is saved in the image directory, by
default a file called ``.meta.db``.

Example usage::

    $ nbc-trainer config.yml meta images/orchids/


.. _nbc-trainer-data:

data
----

Create a tab separated file with training data. :ref:`Preprocessing steps
<config-preprocess>`, :ref:`features to extract <config-features>`, and a
:ref:`classification filter <config-classification.filter>` must be set in a
configurations file.

Example usage::

    $ nbc-trainer config.yml data --cache-dir cache/ \
    > -o train_data.tsv images/orchids/


.. _nbc-trainer-data-batch:

data-batch
----------

In contrast to the :ref:`nbc-trainer-data` subcommand, this will
automatically create all the training data files needed to train neural
networks for classification on each level in a :ref:`classification hierarchy
<config-classification.hierarchy>`. It uses the classification hierarchy to
determine which training data files need to be created.

Example usage::

    $ nbc-trainer config.yml data-batch --cache-dir cache/ \
    > -o train_data/ images/orchids/


.. _nbc-trainer-ann:

ann
---

Train an artificial neural network. Optional training parameters
:ref:`config-ann` can be set in a configurations file.

Example usage::

    $ nbc-trainer config.yml ann -o orchid.ann train_data.tsv


.. _nbc-trainer-ann-batch:

ann-batch
---------

The batch equivalent of the :ref:`nbc-trainer-ann` subcommand, and similar
to the :ref:`nbc-trainer-data-batch` subcommand, in that it automatically
creates all the required artificial neural networks needed for classifying an
image on the levels specified in the :ref:`classification hierarchy
<config-classification.hierarchy>`. Training data required for this subcommand
is created with the :ref:`nbc-trainer-data-batch` subcommand.

Example usage::

    $ nbc-trainer config.yml ann-batch --data train_data/ \
    > -o anns/ images/orchids/


.. _nbc-trainer-test-ann:

test-ann
--------

Test an artificial neural network. If ``--output`` is used, then ``--db`` must
be set and the :ref:`classification filter <config-classification.filter>`
must be set in the configurations file.

.. note::

   Test data has the same format as training data, except that the samples
   should contain data that is new to the neural network.

Example usage::

    $ nbc-trainer config.yml test-ann --ann orchid.ann \
    > --error 0.001 -t test_data.tsv -o test-results.tsv \
    > images/orchids/


.. _nbc-trainer-test-ann-batch:

test-ann-batch
--------------

Test the artificial neural networks for a :ref:`classification hierarchy
<config-classification.hierarchy>`.

.. note::

   Use the :ref:`nbc-trainer-data-batch` subcommand with out-of-sample
   images to create a directory with test data for a classification
   hierarchy.

Example usage::

    $ nbc-trainer config.yml test-ann-batch \
    > -t test_data/ --anns neural_networks/ \
    > -o test-results.tsv images/orchids/


.. _nbc-trainer-classify:

classify
--------

Classify an image using a single neural network. The :ref:`classification
filter <config-classification.filter>` must be set in the configurations file.

Example usage::

    $ nbc-trainer config.yml classify --ann orchid.ann \
    > --imdir images/orchids/ --error 0.001 \
    > images/test/14371998807.jpg


.. _nbc-trainer-validate:

validate
--------

Test the performance of trained neural networks. Performs stratified K-fold
cross validation on the neural networks created from a classification hierarchy.

Example usage::

    $ nbc-trainer config.yml validate --cache-dir cache/ -k4 images/orchids/


.. _nbc-trainer-taxa:

taxa
----

Print the taxon hierarcy for the metadata of an image collection. It can be used
to get the taxon hierarchy for the :ref:`config-classification.taxa`
configuration.

Example usage::

    $ nbc-trainer config.yml taxa images/orchids/


.. _nbc-classify:

nbc-classify
============

Classify digital images using artificial neural networks. Each image is
classified on different levels in a :ref:`classification hierarchy
<config-classification.hierarchy>`, which in this case is a taxonomic
hierarchy.

The neural networks on which this script depends are created with a separate
script, :ref:`nbc-trainer`. See its :ref:`nbc-trainer-data-batch` and
:ref:`nbc-trainer-ann-batch` subcommands for more information.

This script depends on the SQLite database file with metadata for a Flickr
harvested image collection. This database is created by
:ref:`nbc-harvest-images`, which is also responsible for archiving the
images in a local directory.

See the ``--help`` option for usage information.

Example usage::

    $ nbc-classify -v --conf config.yml --imdir images/orchids/ \
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


.. _config.yml: https://github.com/naturalis/nbclassify/blob/master/nbclassify/nbclassify/config.yml
