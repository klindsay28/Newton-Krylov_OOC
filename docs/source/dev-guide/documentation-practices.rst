.. _documentation-practices:

=======================
Documentation Practices
=======================

Documentation for the solver is written using `reStructuredText <https://docutils.sourceforge.io/rst.html>`_ (rst) and is rendered into html using `sphinx <https://www.sphinx-doc.org/en/master/index.html>`_.

A convention used in the rst files for the solver's documentation is to have 1 sentence per line.
One benefit of this convention is that updates to the documentation that change individual sentences in the documentation lead to commit diffs that are confined to line containing the changed sentence.
This is in contrast to conventions that break sentences across lines or have sentences start mid-line.
With these conventions, updates to single sentences tend to have commit diffs that span multiple lines.
Another benefit of this convention is easing searching for phrases in sentences in the documentation, as the phrases will not be split across lines.
Note that this convention typically leads to lines whose length exceeds the line length limit used for python source code.
