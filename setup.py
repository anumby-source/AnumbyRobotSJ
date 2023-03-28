import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE/"README.md").read_text()
VERSION = (HERE/"VERSION").read_text()

setup(
    name = "AnumbyRobotSJ",
    version = VERSION,
    description = "Software du Robot Anumby pour le Service Jeunesse de Bures sur Yvette ",
    long_description = README,
    long_description_content_type = "text/markdown",
    url = "https://github.com/anumby-source/RobotServiceJeunesse2023/tree/main/MasterMind",
    author = "Chris Arnault",
    author_email = "chris.arnault@gmail.com",
    license = "CeCILL-B",
    classifiers = [
        "License :: CeCILL-B Free Software License Agreement (CECILL-B)",
        "Programming Language :: Python :: 3",
    ],
    packages = ["AnumbyMasterMind", "AnumbyFormes", "AnumbyVehicule"],
    include_package_data = True,

    package_data = {"AnumbyFormes": ["data/*.npy",
                                     "dataset/*/*.jpg",
                                     "run/models/*.h5"],
                    "AnumbyVehicule": ["*.jpg"]
                    },

    install_requires = [
        "Pillow == 9.1.0",
        "numpy >= 1.24.2",
        "opencv-python >= 4.7.0",
        "tensorflow >= 2.11.0",
        "keras >= 2.11.0",
        "scikit-learn >= 1.2.2",
        "pandas >= 1.5.3",
        "matplotlib >= 3.7.1"
    ],

    entry_points = {
        "console_scripts": [
            "AnumbyMasterMind = AnumbyMasterMind:__main__.main",
            "AnumbyFormes = AnumbyFormes:__main__.main",
            "AnumbyVehicule = AnumbyVehicule:__main__.main",
        ]
    },
)
