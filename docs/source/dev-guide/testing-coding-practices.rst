.. _testing-coding-practices:

============================
Testing and Coding Practices
============================

-------
Testing
-------

A small amount of Continuous Integration (ci) testing is performed on the solver using `github actions <https://docs.github.com/en/actions>`_.
Testing with github actions currently does the following:

#. Run the source code through `isort <https://pycqa.github.io/isort/>`_ to ensure consistent order of python import statements.
#. Run the source code through `black <https://black.readthedocs.io/en/stable/>`_ to check code style against a particular subset of the python style guide in `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.
#. Run the source code through `flake8 <https://flake8.pycqa.org/en/latest/>`_, which analyzes the code and detects various errors.
#. Run the setup script for the test_problem model, which performs 1 fixed point iteration.
   Compare all generated netCDF4 output to a baseline/reference solution.
#. Run pytest, for unit testing.
#. Run the solver for the test_problem model with ``iage`` turned on.
   Compare the ``iage`` output from the initial fixed point iteration to ``iage`` from the fixed point iteration in step 3, which had the ``phosphorus`` tracer module turned on.
   Compare a subset of the generated netCDF4 output from the first Newton iteration to a baseline/reference solution.
#. Run the solver for the test_problem model with ``dye_decay_{suff}:001:010`` turned on.

Because the test_problem model produces different answers on different platforms, baseline comparisons do not check for equality.
They instead use ``numpy.isclose``.

Tests executed via github actions are performed with python versions 3.7, 3.8, 3.9, 3.10, and 3.11.
Conda is currently unable to create environments with the required packages using earlier versions of python.

The solver has been run for the cime_pop model with the iage and abio_dic_dic14 tracer modules successfully on the NCAR/CISL machine cheyenne using python3.6.

Adding more tests, particularly unit tests and verification of solver output for the cime_pop model, is desirable.

~~~~~~~~~~~~~~~~~~~
Interactive Testing
~~~~~~~~~~~~~~~~~~~

All of the tests performed by github actions can be performed interactively, by running the following commands from the toplevel directory of the repo ``./scripts/ci_short.sh``, ``./scripts/ci_long_iage.sh`` and ``./scripts/ci_long_dye_decay.sh``.

~~~~~~~~~~~~~~~~~~
Pre-commit Testing
~~~~~~~~~~~~~~~~~~

The file ``.pre-commit-config.yaml`` provides for ``black`` and ``flake8``, both mentioned above, to be run prior to a git commit proceeding.
The pre-commit hooks also runs some other checks: trailing-whitespace, end-of-file-fixer, check-docstring-first, check-yaml.

A developer should run the command ``pre-commit install`` to enable these checks.

--------------
Python Version
--------------

The minimal version of python required is 3.6, as f-strings are used.
Note that testing is performed for versions 3.7 and higher as support for 3.6 was ceased on `2021-12-23 <https://devguide.python.org/versions/>`_.
