from setuptools import setup

setup(
    name="pyps",
    version="0.0.27",
    description="Calculate the Stark effect in positronium",
    url="",
    author="ad3ller",
    license="BSD 3-clause",
    packages=["pyps"],
    package_data={"pyps": ["data/*.npy"]},
    install_requires=["numpy>=1.15", "sympy>=1.3", "tqdm>=4.15.0"],
    include_package_data=True,
    zip_safe=False,
)
