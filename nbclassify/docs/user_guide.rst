.. _user-guide:

==========
User Guide
==========

This user guide gives a short introduction to using NBClassify and its
associated scripts. For the API documentation of the NBClassify package, see the
:ref:`api-doc`.

The aim of this project is to use machine learning and pattern recognition, in
particular artificial neural networks (ANNs), to automate image classification.
The development of a generic Python package called ImgPheno_ was started to
implement image feature extraction algorithms in Python. The NBClassify package
implements ImgPheno_ to provide a set of Python scripts for image phenotyping,
hierachical neural network training, and image classification. As a proof-of-
concept, NBClassify's scripts implement this package to test automated
classification of photos of lady slipper orchids, orchids of the subfamily
Cypripedioideae_. With this package images can be classified on genus, section,
and species level. NBClassify is configurable, which means that it can be used
to train neural networks for different classification problems.

In addition to the command-line scripts, NBClassify also comes with a user-
friendly web application for image classification called OrchiD. This web
application comes as a Django_ application and can be found in the subdirectory
``html/orchid/``. OrchiD depends on both NBClassify and ImgPheno_, so they need
to be installed. OrchiD can be installed on any Django_ website. OrchiD allows
users to upload their orchid photos which the web application will try to
identify.


Contents:

.. toctree::
   :maxdepth: 4

   scripts
   config


.. _ImgPheno: https://github.com/naturalis/feature-extraction
.. _Cypripedioideae: http://en.wikipedia.org/wiki/Cypripedioideae
.. _Django: https://www.djangoproject.com/
.. _`GitHub repository`: https://github.com/naturalis/img-classify/
