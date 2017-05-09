NBClassify
----------

This repository contains code and examples that demonstrate the ability for
artificial neural networks to classify images of orchid species. The layout
is as follows:

* `html/orchid/`: A Django application for slipper orchid classification. It
  implements the `nbclassify` package.
* `nbclassify/`: A Python package for image fingerprinting (using the
  [ImgPheno][1] package) and recognition via artificial neural networks.
* `scripts/`: Miscellaneous helper scripts.

[![Build Status](https://travis-ci.org/naturalis/nbclassify.svg?branch=master)](https://travis-ci.org/naturalis/nbclassify)
[![Documentation Status](https://readthedocs.org/projects/nbclassify/badge/?version=latest)](https://readthedocs.org/projects/nbclassify/?badge=latest)

[1]: https://github.com/naturalis/imgpheno

For installation instructions, consult http://github.com/naturalis/puppet-orchid
