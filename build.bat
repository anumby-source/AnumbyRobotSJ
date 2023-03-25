python setup.py sdist bdist_wheel
twine check dist/*
tar tzf dist\AnumbyRobotSJ-1.0.0.tar.gz
twine upload -u chris.arnault --repository-url https://test.pypi.org/legacy/ dist\*


