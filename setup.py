# -*- coding: utf-8 -*-

import setuptools
import pathlib


def get_data_files():

    # Final list of files to return.
    data_files = []
    # List of directories to look into
    data_dirs = ['restore/MatlabCodes/']

    for directory in data_dirs:
        p = pathlib.Path('inpystem/' + directory)

        for file in p.rglob('*'):
            if file.suffix != '.mat' and not file.is_dir():
                data_files += [str(file)[9:]]

    data_files += ['restore/MatlabCodes/ITKrMM/InOut',
                   'restore/MatlabCodes/BPFA/InOut']

    return data_files


def get_long_description():
    # Long description at README
    with open("README.md", "r") as fh:
        long_description = fh.read()
    return long_description


# Load versioning data
version = {}
with open("inpystem/version.py") as fp:
    exec(fp.read(), version)


# Packages that are required for inpystem
install_req = ['numpy',
               'scipy',
               'matplotlib',
               'hyperspy',
               'scikit-learn',
               'scikit-image>=0.16'
               ]


setuptools.setup(
    # Name and version
    #
    name='inpystem',
    version=version['version'],
    package_dir={'inpystem': 'inpystem'},

    # Required installations
    install_requires=install_req,
    packages=['inpystem',
              'inpystem.restore',
              'inpystem.tools',
              'inpystem.tests',
              'inpystem.tests.tools'],
    package_data={
        'inpystem': get_data_files()
        },
    include_package_data=True,
    python_requires='>=3.6',

    # Metadata to display on PyPI
    #
    author=version['author']['name'],
    author_email=version['author']['mail'],
    # Descriptions
    description=version['description'],
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    # Locations
    url=version['url']["Source Code"],
    project_urls=version['url'],
    license=version['license'],
    keywords=version['keywords'],
    platforms=version['platforms'],

    # Classifiers
    #
    classifiers=version['classifiers'],
)
