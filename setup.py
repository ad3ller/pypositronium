from setuptools import setup

setup(name='psfs',
      version='0.0.3',
      description='Calculate the Stark effect in positronium',
      url='',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD 3-clause',
      packages=['psfs'],
      install_requires=[
          'tqdm>=4.15.0'
      ],
      include_package_data=True,
      zip_safe=False)
