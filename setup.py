# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from setuptools import find_packages
from setuptools import setup

version = open("nucleus7/VERSION", "r").read().strip()

setup(
    name='nucleus7',
    packages=find_packages(),
    description='nucleus7 library for deep learning training and inference',
    url="https://github.com/AEV/nucleus7",
    version=version,
    author='Oleksandr Vorobiov',
    author_email='oleksandr.vorobiov@audi.de',
    keywords=['nucleus7', 'deep learning'],
    install_requires=[
        'urllib3==1.22',
        'numpy>=1.15.4',
        'numpydoc>=0.8.0,<0.9.0',
        'tqdm>=4.31.1',
        'matplotlib>=2.2,<3.1',
        'joblib>=0.13.2',
        'dpath>=1.4.2',
        'mlflow>=1.0.0',
        'networkx>=2.3',
    ],
    extras_require={
        'pygraphviz': ['pygraphviz>=1.5'],
    },
    scripts=[
        'bin/nc7-create_dataset_file_list',
        'bin/nc7-create_nucleotide_sample_config',
        'bin/nc7-evaluate_kpi',
        'bin/nc7-extract_data',
        'bin/nc7-get_nucleotide_info',
        'bin/nc7-infer',
        'bin/nc7-train',
        'bin/nc7-visualize_project_dna',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MPL 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    data_files=[('', ['nucleus7/VERSION'])],
)
