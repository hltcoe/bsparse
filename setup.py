import os

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


# from https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "rt") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="bsparse",
    version=get_version("bsparse/__init__.py"),
    author="Andrew Yates",
    author_email="first-then-last@mail-service-by-g.com",
    description="toolkit for creating and searching sparse representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hltcoe/bsparse",
    packages=setuptools.find_packages(),
    install_requires=["ir_datasets", "numpy", "torch", "tqdm", "transformers"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "bsparse=bsparse.cli:__main__",
        ]
    },
)
