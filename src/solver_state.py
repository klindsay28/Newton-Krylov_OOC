"""class for representing the state of an iterative solver"""

import functools
import json
import logging
import os

import numpy as np

from .utils import mkdir_exist_okay


class SolverState:
    """
    class for representing the state of an iterative solver
    """

    def __init__(self, name, workdir, resume=False, rewind=False):
        """initialize solver state"""
        logger = logging.getLogger(__name__)
        logger.debug(
            'SolverState, name="%s", workdir="%s", resume="%r", rewind="%r"',
            name,
            workdir,
            resume,
            rewind,
        )

        # ensure workdir exists
        mkdir_exist_okay(workdir)

        self._name = name
        self._workdir = workdir
        self._state_fname = os.path.join(self._workdir, name + "_state.json")
        self._rewound_step_string = None
        if resume:
            self._read_saved_state()
            self._log_saved_state()
            if rewind:
                self._rewound_step_string = self._saved_state["step_log"].pop()
                logger.info(
                    'rewinding step "%s" for "%s"',
                    self._rewound_step_string,
                    self._name,
                )
        else:
            if rewind:
                msg = "rewind cannot be True if resume is False, name=%s" % self._name
                raise RuntimeError(msg)
            self._saved_state = {"iteration": 0, "step_log": []}
            self.log_step("__init__", per_iteration=False)
            logger.info(
                '"%s" iteration now %d', self._name, self._saved_state["iteration"]
            )

    def get_workdir(self):
        """return value of workdir"""
        return self._workdir

    def get_iteration(self):
        """return value of iteration"""
        return self._saved_state["iteration"]

    def inc_iteration(self):
        """increment iteration, reset step_log"""
        logger = logging.getLogger(__name__)
        logger.debug('name="%s"', self._name)
        self._saved_state["iteration"] += 1
        self.log_step("inc_iteration")
        logger.info('"%s" iteration now %d', self._name, self._saved_state["iteration"])
        return self._saved_state["iteration"]

    def log_step(self, stepval, per_iteration=True):
        """add a step to step_log"""
        logger = logging.getLogger(__name__)
        logger.debug('name="%s"', self._name)
        if not self.step_logged(stepval, per_iteration):
            logger.debug('adding "%s" to step_log', stepval)
            log_string = self._step_log_string(stepval, per_iteration)
            self._saved_state["step_log"].append(log_string)
            self._write_saved_state()
        else:
            logger.debug('"%s" already in step_log', stepval)

    def step_logged(self, stepval, per_iteration=True):
        """has step been logged in the current iteration"""
        log_string = self._step_log_string(stepval, per_iteration)
        return log_string in self._saved_state["step_log"]

    def step_was_rewound(self, stepval, per_iteration=True):
        """does stepval correspond to the step that was rewound during __init__"""
        log_string = self._step_log_string(stepval, per_iteration)
        return (
            False
            if self._rewound_step_string is None
            else log_string == self._rewound_step_string
        )

    def set_value_saved_state(self, key, value):
        """add a key value pair to the saved_state dictionary"""
        self._saved_state[key] = value
        self._write_saved_state()
        # confirm that value can be read back in exactly
        self._read_saved_state()
        if isinstance(value, np.ndarray):
            if not np.array_equal(self._saved_state[key], value):
                msg = "saved_state value not recovered on reread"
                raise RuntimeError(msg)
        else:
            if not self._saved_state[key] == value:
                msg = "saved_state value not recovered on reread"
                raise RuntimeError(msg)

    def get_value_saved_state(self, key):
        """get a value from the saved_state dictionary"""
        return self._saved_state[key]

    def _log_saved_state(self):
        """write saved state of solver to log"""
        logger = logging.getLogger(__name__)
        logger.debug('name="%s"', self._name)
        logger.debug("iteration=%d", self._saved_state["iteration"])
        for step_name in self._saved_state["step_log"]:
            logger.debug('"%s" logged', step_name)

    def _step_log_string(self, stepval, per_iteration):
        """string that gets appended to step_log corresponding to stepval"""
        return "%02d:%s" % (self.get_iteration(), stepval) if per_iteration else stepval

    def _write_saved_state(self):
        """write _saved_state to a JSON file"""
        with open(self._state_fname, mode="w") as fptr:
            json.dump(self._saved_state, fptr, indent=2, cls=NumpyEncoder)

    def _read_saved_state(self):
        """read _saved_state from a JSON file"""
        with open(self._state_fname, mode="r") as fptr:
            self._saved_state = json.load(fptr, object_hook=json_ndarray_decode)


class NumpyEncoder(json.JSONEncoder):
    """
    extend JSONEncoder to handle numpy ndarray's
    https://stackoverflow.com/questions/26646362/nump-array-is-not-json-serializable
    """

    def default(self, o):
        """method called by json.dump, when cls=NumpyEncoder"""
        if isinstance(o, np.ndarray):
            return {"__ndarray__": o.tolist()}
        return json.JSONEncoder.default(self, o)


def json_ndarray_decode(dct):
    """decode __ndarray__ tagged entries"""
    if "__ndarray__" in dct:
        return np.asarray(dct["__ndarray__"])
    return dct


def action_step_log_wrap(step, per_iteration=True, post_exit=False):
    """
    Decorator for wrapping functions with args inside step_logged/log_step checks.
    It is for functions that perform actions and don't return values.
    solver_state is assumed to be a named argument to func.
    step is the string argument getting passed to sover_state methods.
    Formatting using .format is applied to step, using the named arguments of func,
    to enable step to depend on func's arguments.
    """

    def outer_wrapper(func):
        @functools.wraps(func)  # to propagate metadata from func through wrapper
        def inner_wrapper(*args, **kwargs):
            solver_state = kwargs["solver_state"]
            if solver_state is not None:
                if solver_state.step_logged(step.format(**kwargs), per_iteration):
                    return
            func(*args, **kwargs)
            if solver_state is not None:
                solver_state.log_step(step.format(**kwargs), per_iteration)
            if post_exit:
                raise SystemExit

        return inner_wrapper

    return outer_wrapper
