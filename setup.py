from setuptools import setup

setup(name='psfs',
      version='0.0.1.dev',
      description='Calculate the Stark effect in Rydberg positronium',
      url='',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD 3-clause',
      packages=['psfs'],
      install_requires=[
          'attrs>=17.2.0', 'tqdm>=4.15.0'
      ],
      include_package_data=True,
      zip_safe=False)
