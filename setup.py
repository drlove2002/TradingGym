import re

from setuptools import find_packages, setup

version = ""
with open("trading_gym/__init__.py") as f:
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE)
    if match is None or match.group(1) is None:
        raise RuntimeError("version is not set")

    version = match.group(1)

if not version:
    raise RuntimeError("version is not set")

requirements = []
with open("requirements.txt") as f:
    requirements.extend(f.read().splitlines())

readme = ""
with open("README.md") as f:
    readme = f.read()

setup(
    name="trading_gym",
    version=version,
    license="MIT",
    description="A Gym environment for Stock market trading using Reinforcement Learning",
    author="Sudip Roy",
    author_email="sudiproy20yo@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
    long_description=readme,
)
