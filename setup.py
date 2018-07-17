"""A setuptools based setup module for the speckle_pattern package."""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='speckle_pattern',
    version='1.2.1',
    description='Generate print-ready pattern images for DIC applications.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ladisk/speckle_pattern',
    author='Domen Gorjup, Janko Slavič, Miha Boltežar',
    author_email='domen.gorjup@fs.uni-lj.si',
    license='MIT',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.6',
        "License :: OSI Approved :: MIT License",
    ],

    keywords='computer vision dic random speckle pattern',
    packages=[],#find_packages(exclude=[]),
    py_modules = ['speckle_pattern.speckle',],

    install_requires=[
        'matplotlib>=2.0.0',
        'numpy>=1.0.0',
        'scipy>=1.0.0',
        'tqdm>=4.10.0',
        'imageio>=2.2.0',
        'piexif>=1.0.13'
    ],
)