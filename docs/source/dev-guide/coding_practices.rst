================
Coding Practices
================

-------
Testing
-------

A small amount of Continuous Integration (ci) testing is performed on the solver using
`travis-ci <https://travis-ci.com/>`_.
Testing on travis-ci currently does the following:

#. Run the source code through black.
#. Run the setup script for the test_problem model.
#. Run the solver for the test_problem model with ``iage`` turned on.

Adding more tests is desirable.
Testing is perfomed with python3.6 and python3.7.
Conda is currently unable to create environments with the required pakages using
earlier or later versions of python.

The solver has been run for the cime_pop model with the iage and abio_dic_dic14 tracer
modules successfully on the NCAR/CISL machine cheyenne using python3.6.

--------------
Python Version
--------------

The minimal version of python required is 3.2 for python3 or 2.7 for python2, as
``allow_no_value`` is passed to ``configparser.ConfigParser()``.
This argument was introduced in python 3.2 (and 2.7).

The use of multiple context expressions, e.g., in ``gen_precond_jacobian`` in
``newton_fcn_base.py``, requires a minimal version of 3.1 (or 2.7).

Usage of ``.format()`` for formatting strings in ``ann_files_to_mean_file`` and
``mon_files_to_mean_file`` in ``src/utils.py`` requires a minimal version of 3.0 (or 2.6).

The usage of f-strings is under consideration, which would require the minimal version
of python to be 3.6.
Support for 2.7 could be maintained by using `future-fstrings
<https://github.com/asottile/future-fstrings>`_.

------------
Coding Style
------------

Code formatting is controlled by `black <https://black.readthedocs.io/en/stable/>`_.
It is included in the conda environment ``Newton-Krylov_OOC``, and can be invoked by
running the command ``black .`` from the toplevel directory of the repo.
Useful arguments to ``black`` are ``--check`` to see if ``black`` wants to make changes,
and ``--diff`` to see what changes ``black`` wants to make.
Automated invocation of ``black`` on ``git commit`` invocations can be enabled by running
the command ``pre-commit install`` from the toplevel directory of the repo.

Code should be run through pylint, and attention paid to errors and warnings.
The following commands, run from the toplevel directory of the repo, perform this
::

   export PYTHONPATH=models
   find src models -name "*.py" | xargs pylint | less -I
