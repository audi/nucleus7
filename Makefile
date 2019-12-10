.PHONY: test, test-local, test-local-fast
NUCLEUS7_DIR=`pwd`

export PYTHONPATH := ${NUCLEUS7_DIR}:${PYTHONPATH}

test:
	tox

test-local:
	pytest --log-level ERROR --disable-pytest-warnings tests

test-local-fast:
	python3 -m pytest -m "not slow" --log-level ERROR --disable-pytest-warnings tests

