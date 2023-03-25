import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE/"README.md").read_text()

setup(
    name = "AnumbyRobotSJ",
    version = "1.0.0",
    description = "Software du Robot Anumby pour le Service Jeunesse de Bures sur Yvette ",
    long_description = README,
    long_description_content_type = "text/markdown",
    url = "https://github.com/anumby-source/RobotServiceJeunesse2023/tree/main/MasterMind",
    author = "Chris Arnault",
    author_email = "chris.arnault@gmail.com",
    license = "MIT License",
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    packages = ["MasterMind"],
    include_package_data = True,
    install_requires = [],
    entry_points = {
        "console_scripts": ["MasterMind = MasterMind:__main__.main"]
    },
)
