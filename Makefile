init:
	pip install -r requirements.txt

test:
	py.test

html:
	cd doc && make html

clean:
	rm -R dist/
	rm -R build/
	rm -R inpystem.egg-info/
