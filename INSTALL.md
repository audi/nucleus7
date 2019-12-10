Installation
============

- [Python versions](#python-version)
- [Package installation](#package-installation)
- [Local installation](#local-installation)
- [Tensorrt installation](#tensorrt-installation)
- [Build documentation](#api-documentation)

[requirements]: ./requirements.txt
[requirements_dev]: ./requirements_dev.txt
[requirements_doc]: ./requirements_docs.txt
[CONTRIBUTING]: ./CONTRIBUTING.md
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/
[sphinx-install]: https://www.sphinx-doc.org/en/master/usage/installation.html
[tensorrt]: https://developer.nvidia.com/tensorrt
[tensorflow-gpu-install]: https://www.tensorflow.org/install/gpu
[Inferer]: ./nucleus7/coordinator/README.md

## Python versions <a name="python-version"></a>

Currently `python 3.5, 3.6 and 3.7`.
If you have issues or need other interpreter version,
[raise a Feature Request or report a bug][CONTRIBUTING]

## Package installation <a name="package-installation"></a>

To install it, just type (remove virtual environment commands if you want to
install in the default environment):

```bash
virtualenv --python=python3.5 ~/.env-nucleus7
source ~/.env-nucleus7/bin/activate
python3 setup.py install
```

You are free to select the tensorflow version, but it was tested with
tensorflow(-gpu) >=1.11, <1.15

## Local installation <a name="local-installation"></a>

First you need to install the [requirements][requirements]:

```bash
pip3 install -r requirements.txt
```

or [dev requirements][requirements_dev] (includes requirements for testing):

```bash
pip3 install -r requirements_dev.txt
```

or [documentation requirements][requirements_doc] (includes requirements for docs build):

```bash
pip3 install -r requirements_docs.txt
```

If you want to install all the extra dependencies, that are needed for some
features, use:

```bash
pip3 install -r requirements_extras.txt
```

To setup the path and get a number of shortcut aliases (all start with `nc7-`)
you can run the `activate.sh` script which will setup everything inside the
current shell session:

```bash
source activate.sh
```

## Tensorrt installation <a name="tensorrt-installation"></a>

If you want to accelerate your inference using [tensorrt][tensorrt], you
can install it as described inside of
[tensorflow gpu install][tensorflow-gpu-install] or check the nvidia
link. After you can activate it as described in
[inferer manual][Inferer]

## Build documentation <a name="api-documentation"></a>

Most important methods and classes are documented inside of its docstring using
[NumpyDoc][numpydoc] format. So we can build
the documentation using sphinx.
So first [install sphinx][sphinx-install] if you don't have it,
and the build the docs:

```bash
cd docs
make html
```
