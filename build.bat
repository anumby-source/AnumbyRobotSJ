set /P VERSION=<VERSION
echo "version = %VERSION%"
set PACKAGE=AnumbyRobotSJ
REM rmdir dist /s/q
python setup.py sdist bdist_wheel
twine check dist/%PACKAGE%-%VERSION%-py3-none-any.whl
tar tzf dist\%PACKAGE%-%VERSION%.tar.gz
twine upload -u chris.arnault --repository-url https://test.pypi.org/legacy/ dist\%PACKAGE%-%VERSION%*.*
REM pip install -i https://test.pypi.org/simple/ AnumbyRobotSJ==%VERSION%
REM twine upload -u chris.arnault dist\%PACKAGE%-%VERSION%*.*
