from setuptools import setup, find_packages

__version__ = "0.0.1"

INSTALL_REQUIRES = ['tensorflow']

setup(name='mkv-google-bert',
      version=__version__,
      description='Package for training and using Google BERT model.',
      author='Matej Kvassay',
      author_email='matejkvassay@icloud.com',
      packages=find_packages(),
      namespace_packages=['mkv'],
      install_requires=INSTALL_REQUIRES,
      )
