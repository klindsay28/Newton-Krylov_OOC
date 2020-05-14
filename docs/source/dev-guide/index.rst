.. _dev-guide:

===================================
Newton-Krylov_OOC Developer's Guide
===================================

--------------
Python Version
--------------

The minimal version of python is 3.0, as there is some usage of python3 constructs.
For example, ``.format()`` is applied to strings in ``src/utils.py``.

The usage of f-strings is under consideration, which would require the minimal version
of python to be 3.6.

------------
Coding Style
------------

Code formatting is controlled by `black <https://black.readthedocs.io/en/stable/>`_.
It is included in the conda environment ``Newton-Krylov_OOC``, and can be invoked by
running the command ``black .`` from the root directory of the repo.
Useful arguments to ``black`` are ``--check`` to see if ``black`` wants to make changes,
and ``--diff`` to see what changes ``black`` wants to make.
Automated invocation of ``black`` on ``git commit`` invocations can be enabled by running
the command ``pre-commit install`` from the root directory of the repo.
