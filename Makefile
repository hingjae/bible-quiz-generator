build-zip:
	rm -rf ./python layer.zip
	pip install -r ./app/requirements.txt --platform manylinux2014_x86_64 --target ./python --only-binary=:all:
	zip -r layer.zip python/
