#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

import re
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

extra_include_dirs = ['.']
try:
    import numpy
    extra_include_dirs += [numpy.get_include()]
except:
    pass

ext_args = dict(
    include_dirs=extra_include_dirs,
    extra_compile_args=['-Ofast'],
    extra_link_args=['-Ofast'],
)


with open('README.rst', encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding="utf-8") as history_file:
    history = re.sub(r':py:class:`([^`]+)`', r'\1', 
        history_file.read())
    

requirements = ['numpy', 'cython', 'ultranest']

setup_requirements = ['pytest-runner', ]
test_requirements = ['pytest>=3', ]

setup(
    author="Johannes Buchner",
    author_email='johannes.buchner.acad@gmx.com',
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Fast X-ray Spectral Fitting",
    install_requires=requirements,
    ext_modules = cythonize([
        Extension('fastxsf.response_helper', ["fastxsf/response_helper.pyx"], 
            **ext_args),
    ]),
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fastxsf',
    name='fastxsf',
    packages=['fastxsf'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JohannesBuchner/fastxsf',
    version='0.0.1',
    zip_safe=False,
    cmdclass={'build_ext': build_ext},
)
