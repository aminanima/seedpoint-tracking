import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = read('README.rst')

VERSION = find_version('seedpoint_tracking', '__init__.py')

requirements = ['enum34']

setup(
    # Metadata
    name='seedpoint-tracking',
    version=VERSION,
    author='Jose Caballero',
    author_email='jcaballero@twitter.com',
    url='https://github.com/josecabjim/seedpoint-tracking',
    description='A library for video object tracking from a seed point.',
    long_description=readme,
    license='Apache 2',

    # Package info
    packages=find_packages(exclude=('tests',)),

    zip_safe=True,
    install_requires=requirements,
)
