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
	find inpystem/ -type d -name '__pycache__' -exec rm -R {} \;
