# MasterMind

Ce package Python est disponible en public. Il s'installe avec l'outil `pip` (il faut avoir Python installé). Une fois installé (https://github.com/anumby-source/AnumbyRobotSJ#installation)
vous disposez de plusieurs applications:

- [AnumbyFormes](#anumbyformes)
- [AnumbyVehicule](#anumbyvehicule)

et bientôt...

- AnumbyMasterMind

## AnumbyFormes

- construit les figures pour le jeu MasterMind
- produit des données d'apprentissage pour le réseau de neurones artificiel (Kéras)
- entraîne le RdN sur ces données


- `AnumbyFormes -h`
- `AnumbyFormes -figures [-f <numéro>] [-cell <int=40>] [-formes <int=8>]` construit les figures de base
- `AnumbyFormes -run [-build_data] [-data <int=100>]` reconstruit les données d'entraînement et lance l'apprentissage
- `AnumbyFormes -run [-build_model]` reconstruit le modèle et lance l'apprentissage sur des données existantes

## AnumbyVehicule

- Simule l'installation d'un véhicule robotisé pour opérer un jeu de MasterMind avec une caméra capable de reconnaître les figures

`AnumbyVehicule`

![Ecran](Ecran.GIF)

![Contrôle](Contrôle.GIF)

## Installation

``pip install AnumbyRobotSJ``

ou

``pip install AnumbyRobotSJ==<version>``

## Déinstallation

``pip uninstall -y AnumbyRobotSJ``

# Reconstruction du package.

- 1) incrémenter le numéro de version => modifier VERSION
- 2) lancer `build.bat`

# License

Copyright 2023 Chris Arnault

License CECILL
