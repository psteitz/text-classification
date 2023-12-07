from setuptools import setup, find_packages

setup(
    name='text-classification',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='Apache 2.0',
    description='Text classification using transformer-based models.',
    long_description=open('README.md').read(),
    install_requires=[],
    url='https://github.com/psteitz/text-classification',
    author='Phil Steitz',
    author_email='phil@steitz.com'
)
