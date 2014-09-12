.. _user-guide:

==========
User Guide
==========

This user guide attempts to give a short introduction to using NBClassify and
its associated scripts. For a detailed documentation of the NBClassify package
contents, see the :ref:`api-doc`.

The aim of this project is to leverage artificial intelligence, in particular
artificial neural networks (ANNs), to automate image classification. The
development of a generic Python package called ImgPheno_ was started to
implement image fingerprinting in Python. The NBClassify package implements
ImgPheno_ to provide a set of Python scripts for image fingerprinting, neural
network training, and image classification. As a proof-of-concept, NBClassify
implements this package to test automated classification of photos of lady's
slipper orchids, orchids of the subfamily Cypripedioideae_. With this package,
images can be classified on genus, section, and species level.

In addition to the command-line scripts, NBClassify also comes with a more
user-friendly web application for image classification called OrchiD. This web
application comes as a Django_ application, and can be found in the `GitHub
repository`_ in the subdirectory ``html/orchid/``. OrchiD requires that both
NBClassify and ImgPheno_ are installed. OrchiD can be easily installed on any
Django_ website.


Using the scripts
=================

NBClassify comes with several command-line scripts. Usage of these scripts is
explained in the following subsections:

.. toctree::
   :maxdepth: 4

   scripts


.. _ImgPheno: https://github.com/naturalis/feature-extraction
.. _Cypripedioideae: http://en.wikipedia.org/wiki/Cypripedioideae
.. _Django: https://www.djangoproject.com/
.. _`GitHub repository`: https://github.com/naturalis/img-classify/