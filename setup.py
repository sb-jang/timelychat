from setuptools import find_packages, setup

setup(
    name="python-39-template-repository",
    version="0.0.1",
    description="python template repository",
    install_requires=[],
    url="https://github.com/sb-jang/python-39-template.git",
    author="Seongbo Jang",
    author_email="jang.sb@postech.ac.kr",
    packages=find_packages(exclude=["tests"]),
)
