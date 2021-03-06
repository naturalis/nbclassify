from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nbclassify',
    version='0.1.1',
    description='Train artificial neural networks and image classification',
    long_description=long_description,
    url='https://github.com/naturalis/nbclassify',
    author='Naturalis Biodiversity Center',
    author_email='serrano.pereira@naturalis.nl',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Environment :: Console'
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='opencv numpy fann image recognition computer vision',
    packages=find_packages(exclude=['docs', 'scripts']),
    install_requires=[
        'flickrapi',
        'imgpheno',
        'numpy',
        'scikit-learn>=0.15',
        'sqlalchemy>=0.9.1',
        'pyyaml'
    ],
    package_data={
        'nbclassify': ['config.yml', 'config_aivolver.yml'],
    },
    scripts=[
        'scripts/nbc-classify',
        'scripts/nbc-harvest-images',
        'scripts/nbc-trainer'
    ]
)
