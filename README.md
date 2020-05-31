# Out-of-Core Newton-Krylov Solver

[![Build Status](https://travis-ci.com/klindsay28/Newton-Krylov_OOC.svg?branch=master)](https://travis-ci.com/klindsay28/Newton-Krylov_OOC)
[![Documentation Status](https://readthedocs.org/projects/newton-krylov-ooc/badge/?version=latest)](https://newton-krylov-ooc.readthedocs.io/en/latest/?badge=latest)

## Documentation

https://newton-krylov-ooc.readthedocs.io/

## Background

A Newton-Krylov (NK) solver is a nested iterative solver for approximating the solution
of a system of equations.
An application in the realm of time-dependent ordinary or partial differential equations
is to find an initial condition such that the solution of the differential equations,
with that initial condition, is cyclo-stationary (periodic) solution in time.

Each iteration of a NK solver requires the evaluation of the function being solved for.
For large systems of equations, this evaluation can take hours of wall-clock time on a
supercomputer, and may require utilizing a batch job submission system.
Typical implementations of NK solvers are in-core.
They have a driver program that stays resident in memory while function evaluations
occur.
In some computing environments, it is impractical, or forbidden, to have such a driver
program reside in memory for the amount of time needed for multiple function
evaluations.
A primary goal of the NK solver implemented in this repository is to avoid this in-core
requirement.

## High-level Solver Description

The solver tracks solver-state, which includes a list of steps that the solver has
completed and computational results of the solver.
The solver-state is saved to a file whenever the solver-state is updated. The solver has
a resume option.
When specified, the solver reads in a previously saved solver-state and skips previously
completed steps.
When function evaluation invokes a long running batch job, the solver exits after the
job is submitted, and the batch job is augmented to invoke the solver with the resume
option specified.
This implementation effectively enables the solver to resume where it left off, and
avoids the requirement for the solver to remain resident in memory.

## Directory Hierarchy Sketch
(not complete or sorted)
```
.
    README.md
    ...

    docs/               # solver documentation

    src/                # directory for python code applicable to multiple models
        nk_driver.py
        model_state_base.py
        utils.py
        ...
        test_problem/   # subdirectory for python code that is model specific
        cime_pop/       # subdirectory for python code that is model specific

    scripts/            # directory for scripts that are applicable to multiple models
                        #   (there are no model specific scripts)
    input/           
        test_problem/   # directory for input files that are model specific
        cime_pop/       # directory for input files that are model specific
```
