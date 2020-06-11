============================
Testing and Coding Practices
============================

-------
Testing
-------

A small amount of Continuous Integration (ci) testing is performed on the solver using `travis-ci <https://travis-ci.com/>`_.
Testing on travis-ci currently does the following:

#. Run the source code through `black <https://black.readthedocs.io/en/stable/>`_ to check code style against a particular subset of the python style guide in `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.
#. Run the source code through `flake8 <https://flake8.pycqa.org/en/latest/>`_, which analyzes the code and detects various errors.
#. Run the setup script for the test_problem model, which performs 1 forward model run.
#. Run pytest, for unit testing.
#. Run the solver for the test_problem model with ``iage`` turned on.
#. Run the solver for the test_problem model with ``dye_decay_{suff}:001:010`` turned on.

Current testing does not verify the output of running the solver, the testing just confirms that the solver runs without generating an error.

Adding more tests, particularly unit tests and verification of solver output, is desirable.

Tests executed via travis-ci are performed with python versions 3.6, 3.7, and 3.8.
Conda is currently unable to create environments with the required pakages using earlier versions of python.

The solver has been run for the cime_pop model with the iage and abio_dic_dic14 tracer modules successfully on the NCAR/CISL machine cheyenne using python3.6.

~~~~~~~~~~~~~~~~~~~
Interactive Testing
~~~~~~~~~~~~~~~~~~~

All of the tests performed by travis-ci can be performed interactively, by running the following commands from the toplevel directory of the repo ``./scripts/travis_short.sh``, ``./scripts/travis_long_iage.sh`` and ``./scripts/travis_long_dye_decay.sh``.

~~~~~~~~~~~~~~~~~~
Pre-commit Testing
~~~~~~~~~~~~~~~~~~

The file ``.pre-commit-config.yaml`` provides for ``black`` and ``flake8``, both mentioned above, to be run prior to a git commit proceeding.
The pre-commit hooks also runs some other checks: trailing-whitespace, end-of-file-fixer, check-docstring-first, check-yaml.

A developer should run the command ``pre-commit install`` to enable these checks.

--------------
Python Version
--------------

The minimal version of python required is 3.2 for python3 or 2.7 for python2, as ``allow_no_value`` is passed to ``configparser.ConfigParser()``.
This argument was introduced in python 3.2 (and 2.7).

The use of multiple context expressions, e.g., in ``gen_precond_jacobian`` in ``newton_fcn_base.py``, requires a minimal version of 3.1 (or 2.7).

Usage of ``.format()`` for formatting strings in ``ann_files_to_mean_file`` and ``mon_files_to_mean_file`` in ``src/utils.py`` requires a minimal version of 3.0 (or 2.6).

The usage of f-strings is under consideration, which would require the minimal version of python to be 3.6.
Support for 2.7 could be maintained by using `future-fstrings <https://github.com/asottile/future-fstrings>`_.
