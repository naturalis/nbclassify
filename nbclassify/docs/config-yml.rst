.. _config-yml:

===================
Configurations file
===================

This document describes the configurations file that is used by the scripts
:ref:`nbc-trainer-py` and :ref:`nbc-classify-py`. This configurations file is
in the YAML_ format and specifies what preprocessing needs to be done on
images, what features need to be extracted from the images, what the
classification hierarchy looks like, and how the neural networks are trained.
See :download:`nbclassify/config.yml <../nbclassify/config.yml>` for an
example configurations file.

The following sections exists in this configurations file:

.. _config-yml-classification-hierarchy:

classification.hierarchy
========================

A classification hierarchy consists of `levels`, and each level has a name. In
the case of the slipper orchids, the levels correspond to the taxanomic ranks:
genus, section, species. In this case, this means that each image can be
classified on three levels. This is what a classification hierarchy looks
like:

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
classified on the genus level. Once the genus is known, it will be classified
on section level within that specific genus. And when the section is know, it
will be classified on species level within that section. The
``{train|test|ann}_file`` settings support ``__level__`` wildcards, where
``level`` can be the name of any parent level. During classification, the
``__level__`` wildcards are replaced by the correspondings level
classification made in the parent levels.

The classification hierarcy supports the following settings:

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


.. _YAML: http://yaml.org/
