init:
	pip install -r requirements.txt

test:
	py.test tests

html:
	cd doc && make html

clean:
	rm -R dist/
	rm -R build/
	rm -R pystem.egg-info/
