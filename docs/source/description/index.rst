=============================
Newton-Krylov_OOC Description
=============================

==========
Motivation
==========

The application that motivated the development of Newton-Krylov_OOC is
spinning up biogeochemical tracers in an ocean general circulation model (OGCM).
Literature on the topic demonstrated that Newton-Krylov based solvers
could be used effectively to solve this problem
:cite:`Li_Primeau_OceMod_2008,Khatiwala_OceMod_2008`.
An excellent book on Newton-Krylov solvers, :cite:`Kelley_Newtons_Method`,
advises that is is preferable to use an existing Newton–Krylov
implementation instead of implementing your own.
However, existing implementations of Newton-Krylov solvers have some
features that make their usage impractical.

.. bibliography:: ../references.bib
