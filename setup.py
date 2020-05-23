from setuptools import setup

setup(name='pyps',
      version='0.0.10',
      description='Calculate the Stark effect in positronium',
      url='',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD 3-clause',
      packages=['pyps'],
      install_requires=[
          'numpy>=1.15', 'sympy>=1.3', 'tqdm>=4.15.0'
      ],
      include_package_data=True,
      zip_safe=False)
