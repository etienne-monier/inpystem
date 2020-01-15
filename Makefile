.PHONY: init test html build clean

init:
	pip install -r requirements.txt

test:
	py.test

html:
	cd docs && make html

build:
	python setup.py sdist bdist_wheel
	twine check dist/*
	# cf 
	# https://realpython.com/pypi-publish-python-package/#different-ways-of-calling-a-package
	# https://packaging.python.org/tutorials/packaging-projects/
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

clean:
	rm -R dist/
	rm -R build/
	rm -R inpystem.egg-info/
