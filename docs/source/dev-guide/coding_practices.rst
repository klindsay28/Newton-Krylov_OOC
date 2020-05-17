================
Coding Practices
================

--------------
Python Version
--------------

Development is currently done using python3.6.
Testing with earlier versions of python has not been `implemented
<https://github.com/klindsay28/Newton-Krylov_OOC/issues/15>`_.
Until that testing is implemented, and results are verified, support for earlier versions
of python is aspirational.

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
