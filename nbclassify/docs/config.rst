.. highlight:: yaml

.. _config:

=============
Configuration
=============

This document describes the configurations file that is used by the scripts
:ref:`nbc-trainer` and :ref:`nbc-classify`. This configurations file is
in the YAML_ format and specifies what preprocessing needs to be done on
images, what features need to be extracted from the images, what the
classification hierarchy looks like, and how the neural networks are trained.
See config.yml_ for an example configurations file.

The following settings exists in this configurations file:

.. _config-directory_hierarchy:

directory_hierarchy
===================

The directory hierarchy describes how images are organized in an image directory
on the local hard drive, and is needed for :ref:`compiling the metadata file
<nbc-trainer-meta>` for an image directory. Each item in the list corresponds to
a taxonomic rank. The directory name will be stored as the taxonomic name for
that rank. If the directory name is "None", that rank will not be saved in the
database. Directory levels can be ignored by setting the corresponding rank in
the directory hierarchy to ``__ignore__``.

Example::

    directory_hierarchy:
        - genus
        - section
        - species


.. _config-data:

data
====

Describe training data format. The file format for training data is tab
separated. The header row describes the columns in the data file. A minimal
training data file contains columns for input data (determinants), and columns
for output data (dependents). An optional third column type with the name "ID"
may exists, which contains labels for the samples (data rows in the training
file).


.. _config-data.dependent_prefix:

data.dependent_prefix
---------------------

Columns for output data are recognized by the NBClassify scripts by a prefix
in the column names. With this setting the prefix for output columns can be
specified. It is used for exporting, as well as reading training data. When
training data is exported, the prefix is extended with a number.

Example::

    data:
        dependent_prefix: "OUT:"


.. _config-preprocess:

preprocess
==========

Preprocessing steps to perform on images before features are extracted.


.. _config-preprocess.maximum_perimeter:

preprocess.maximum_perimeter
----------------------------

Limit the maximum perimeter for input images (optional). The input image is
scaled down to the maximum perimeter if its perimeter, calculated as ``width +
height``, exeeds this value. The input images on disk stay unmodified.

Example::

    preprocess:
        maximum_perimeter: 1000


.. _config-preprocess.color_enhancement.naik_murthy_linear:

preprocess.color_enhancement.naik_murthy_linear
-----------------------------------------------

Hue-preserving color image enhancement. Provides a hue preserving linear
transformation with maximum possible contrast. [1]_ This works best on images
with at least one pixel that should be black, and a pixel that should be
white.

Example::

    color_enhancement:
        naik_murthy_linear: {}


.. _config-preprocess.segmentation.grabcut:

preprocess.segmentation.grabcut
-------------------------------

Runs the GrabCut algorithm [2]_ for foreground segmentation. The GrabCut
algorithm is executed with `iters` iterations. If the region of interest is
not set, the region of interest is set to the entire image, with a margin of
`margin` pixels from the image boundaries.

Example::

    preprocess:
        segmentation:
            grabcut:
                iters: 5
                margin: 1

The following options are available:

iters
  The number of segmentation iterations. Defaults to 5.

margin
  The margin of the region of interest from the edges of the image. Defaults
  to 1.


.. _config-features:

features
========

Features to be extracted from objects in images.


.. _config-features.color_bgr_means:

features.color_bgr_means
------------------------

Describes the BGR color frequencies along horizontal and vertical axis. Each
axis is divided into equal size bins. The mean B, G, and R are computed for
each bin.

Example::

    features:
        color_bgr_means:
            bins: 50

The following options are available:

bins
  The number of bins to use for each axis. Defaults to 20.


.. _config-features.shape_outline:

features.shape_outline
------------------------

Describes the shape outline. The shape is measured along `k` points on both X
and Y axis.

Example::

    features:
        shape_outline:
            k: 15

The following options are available:

k
  The shape is measured on `k` points on both X and Y axis.


.. _config-features.shape_360:

features.shape_360
------------------

Describes the shape in 360 degrees.

.. note::

   This feature is experimental and does not work well with small images.
   It may even fail to extract a shape feature from small images.

Example::

    features:
        shape_360:
            rotation: 0
            step: 1
            t: 8
            output_functions:
                mean_sd: {}

The following options are available:

rotation
  Specify rotation if the object is rotated (default is 0, no rotation). Set
  to ``FIT_ELLIPSE`` to automatically get the rotation for each image by
  ellipse fitting. Rotations up to 90 degrees means rotation to the right.
  Rotations 91 to 179 means rotation to the left (e.g. 95 equals 5 degrees to
  the left).

step
  Step size for the 360 angles. If set to 1, then all 360 angles are measured.
  When set to 2, every other angle is measured, etc. Defaults to 1.

t
  Distance threshold in pixels for point clustering. Defaults to 8.

output_functions
  The output functions control how the shape is returned. Multiple output
  functions can be specified:

  * ``mean_sd: {}``: Returns the mean length and standard deviation of the
    vector from the object center to all outline intersections for each angle.
    This is the default output function.


.. _config-ann:

ann
===

Parameters for training artificial neural networks.

Example::

    ann:
        train_type: ordinary
        desired_error: 0.00001
        training_algorithm: TRAIN_RPROP
        activation_function_hidden: SIGMOID_SYMMETRIC
        activation_function_output: SIGMOID_SYMMETRIC

        # Ordinary training:
        epochs: 100000
        hidden_layers: 1
        hidden_neurons: 20
        learning_rate: 0.7
        connection_rate: 1

        # Cascade training:
        max_neurons: 100
        neurons_between_reports: 1
        cascade_activation_steepnesses: [ 0.25, 0.50, 0.75, 1.00 ]
        cascade_num_candidate_groups: 2

The following options are available:

train_type
  Training type: ``ordinary`` or ``cascade`` training. Defaults to ``ordinary``.

desired_error
  Desired error. Defaults to 0.00001.

training_algorithm
  The training algorithm used for training. Defaults to ``TRAIN_RPROP``. See
  fann_train_enum_.

activation_function_hidden
  The activation function for the hidden layers. Defaults to
  ``SIGMOID_STEPWISE``. See fann_activationfunc_enum_.

activation_function_output
  The activation function for the output layer. Defaults to
  ``SIGMOID_STEPWISE``. See fann_activationfunc_enum_.

Options for ordinary training:

epochs
  Maximum number of epochs. Defaults to 100000.

hidden_layers
  Number of hidden neuron layers. Defaults to 1.

hidden_neurons
  Number of hidden neurons per hidden layer. Defaults to 8.

learning_rate
  Learning rate. Defaults to 0.7. See fann_get_learning_rate_.

connection_rate
  Connection rate. Defaults to 1, a fully connected network. See
  fann_create_sparse_.

Options for cascade training:

max_neurons
  The maximum number of neurons to be added to neural network. Defaults to 20.

neurons_between_reports
  The number of neurons between printing a status report. Defaults to 1.

cascade_activation_steepnesses
  List of the different activation functions used by the candidates. Defaults to
  ``[ 0.25, 0.50, 0.75, 1.00 ]``. See fann_get_cascade_activation_steepnesses_.

cascade_num_candidate_groups
  The number of groups of identical candidates which will be used during
  training. Defaults to 2. See fann_set_cascade_num_candidate_groups_.

.. _config-classification:

classification
==============

Configurations for classification.

.. _config-classification.filter:

classification.filter
----------------------

Simplified query for selecting images with corresponding classification from
the meta data database.

An example that selects all images of genus *Cypripedium* and section
*Arietinum*, classifying by species::

    classification:
        filter:
            where:
                genus: Cypripedium
                section: Arietinum
            class:
                species

A filter has the following keys:

where
  A set of rank:taxon pairs used to filter images by. Each key corresponds to a
  taxonomic rank set for an image, and the value is a taxon name for that
  rank.

class
  Specifies which rank to use as the classification for an image.


.. _config-classification.hierarchy:

classification.hierarchy
------------------------

A classification hierarchy consists of `levels`, and each level has a name. In
the case of the slipper orchids, the levels correspond to the taxanomic ranks:
genus, section, species. In this case, this means that each image can be
classified on three levels. This is what a classification hierarchy looks
like::

    classification:
        hierarchy:
            - name: genus
              data: *data
              preprocess: *preprocess_std
              features: *features_std
              ann: *ann_genus
              train_file: genus.tsv
              test_file: genus.tsv
              ann_file: genus.ann
              max_error: 0.00001

            - name: section
              data: *data
              preprocess: *preprocess_std
              features: *features_std
              ann: *ann_std
              train_file: __genus__.section.tsv
              test_file: __genus__.section.tsv
              ann_file: __genus__.section.ann
              max_error: 0.0001

            - name: species
              data: *data
              preprocess: *preprocess_std
              features: *features_std
              ann: *ann_std
              train_file: __genus__.__section__.species.tsv
              test_file: __genus__.__section__.species.tsv
              ann_file: __genus__.__section__.species.ann
              max_error: 0.001

The order of the levels is important. In this example, each image is first
classified on the genus level. Once the genus is known, it will be classified
on section level within that specific genus. And when the section is know, it
will be classified on species level within that section. The
``{train|test|ann}_file`` settings support ``__level__`` wildcards, where
``level`` can be the name of any parent level. During classification, the
``__level__`` wildcards are replaced by the correspondings level
classification made in the parent levels.

The following options are available:

name*
  The name of the level.

data
  Training data format.

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


.. _config-classification.taxa:

classification.taxa
-------------------

The taxon hierarchy to be used for classification. This must be the same taxon
hierarchy used while training the set of artificial neural networks being used
for classification. Setting this option allows image classification without the
need for a metadata database.

The taxon hierarchy for the metadata of an image collection can be obtained with
the :ref:`nbc-trainer-taxa` subcommand of the train script.

----

.. [1] Naik, S. K. & Murthy, C. A. Hue-preserving color image enhancement
       without gamut problem. IEEE Trans. Image Process. 12, 1591–8 (2003).

.. [2] Rother, C., Kolmogorov, V. & Blake, A. GrabCut — Interactive Foreground
       Extraction using Iterated Graph Cuts. ACM Trans. Graph. (2004). at
       <http://research.microsoft.com/apps/pubs/default.aspx?id=67890>

.. _YAML: http://yaml.org/
.. _config.yml: https://github.com/naturalis/img-classify/blob/master/nbclassify/nbclassify/config.yml
.. _GrabCut:
.. _fann_get_learning_rate: http://leenissen.dk/fann/html/files/fann_train-h.html#fann_get_learning_rate
.. _fann_create_sparse: http://leenissen.dk/fann/html/files/fann-h.html#fann_create_sparse
.. _fann_train_enum: http://leenissen.dk/fann/html/files/fann_data-h.html#fann_train_enum
.. _fann_activationfunc_enum: http://leenissen.dk/fann/html/files/fann_data-h.html#fann_activationfunc_enum
.. _fann_get_cascade_activation_steepnesses: http://leenissen.dk/fann/html/files/fann_cascade-h.html#fann_get_cascade_activation_steepnesses
.. _fann_set_cascade_num_candidate_groups: http://leenissen.dk/fann/html/files/fann_cascade-h.html#fann_set_cascade_num_candidate_groups
