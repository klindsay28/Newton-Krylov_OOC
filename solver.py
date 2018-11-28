"""class for representing the state of an iterative solver"""

import json
import logging
import os

import numpy as np

import util

class SolverState:
    """
    class for representing the state of an iterative solver
    """

    def __init__(self, name, workdir, resume=False, rewind=False):
        """initialize solver state"""
        logger = logging.getLogger(__name__)
        logger.debug('SolverState:entering, name="%s"', name)

        # ensure workdir exists
        util.mkdir_exist_okay(workdir)

        self._name = name
        self._workdir = workdir
        self._state_fname = os.path.join(self._workdir, name+'_state.json')
        self._rewound_step_string = None
        if resume:
            self._read_saved_state()
            self._log_saved_state()
            if rewind:
                self._rewound_step_string = self._saved_state['step_log'].pop()
                logger.info('rewinding step %s for %s', self._rewound_step_string, self._name)
        else:
            if rewind:
                msg = 'rewind cannot be True if resume is False, name=%s' % self._name
                raise RuntimeError(msg)
            self._saved_state = {'iteration':0, 'step_log':[]}
            self.log_step('__init__')
            logger.info('%s iteration now %d', self._name, self._saved_state['iteration'])

        logger.debug('returning')

    def get_workdir(self):
        """return value of workdir"""
        return self._workdir

    def get_iteration(self):
        """return value of iteration"""
        return self._saved_state['iteration']

    def inc_iteration(self):
        """increment iteration, reset step_log"""
        logger = logging.getLogger(__name__)
        logger.debug('entering, name="%s"', self._name)
        self._saved_state['iteration'] += 1
        self.log_step('inc_iteration')
        logger.info('%s iteration now %d', self._name, self._saved_state['iteration'])
        logger.debug('returning')
        return self._saved_state['iteration']

    def log_step(self, stepval):
        """add a step to step_log"""
        logger = logging.getLogger(__name__)
        logger.debug('entering, name="%s"', self._name)
        if not self.step_logged(stepval):
            logger.debug('adding "%s" to step_log', stepval)
            self._saved_state['step_log'].append(self._step_log_string(stepval))
            self._write_saved_state()
        else:
            logger.debug('"%s" already in step_log', stepval)
        logger.debug('returning')

    def step_logged(self, stepval):
        """has step been logged in the current iteration"""
        return self._step_log_string(stepval) in self._saved_state['step_log']

    def step_was_rewound(self, stepval):
        """does stepval correspond to the step that was rewound during __init__"""
        return False if self._rewound_step_string is None \
            else self._step_log_string(stepval) == self._rewound_step_string

    def set_value_saved_state(self, key, value):
        """add a key value pair to the saved_state dictionary"""
        self._saved_state[key] = value
        self._write_saved_state()
        # confirm that value can be read back in exactly
        self._read_saved_state()
        if isinstance(value, np.ndarray):
            if not np.array_equal(self._saved_state[key], value):
                raise RuntimeError('saved_state value not recovered on reread')
        else:
            if not self._saved_state[key] == value:
                raise RuntimeError('saved_state value not recovered on reread')

    def get_value_saved_state(self, key):
        """get a value from the saved_state dictionary"""
        return self._saved_state[key]

    def _log_saved_state(self):
        """write saved state of solver to log"""
        logger = logging.getLogger(__name__)
        logger.debug('name=%s', self._name)
        logger.debug('iteration=%d', self._saved_state['iteration'])
        for step_name in self._saved_state['step_log']:
            logger.debug('%s logged', step_name)

    def _step_log_string(self, stepval):
        """string that gets appended to step_log corresponding to stepval"""
        return '%02d:%s' % (self.get_iteration(), stepval)

    def _write_saved_state(self):
        """write _saved_state to a JSON file"""
        with open(self._state_fname, mode='w') as fptr:
            json.dump(self._saved_state, fptr, indent=2, cls=NumpyEncoder)

    def _read_saved_state(self):
        """read _saved_state from a JSON file"""
        with open(self._state_fname, mode='r') as fptr:
            self._saved_state = json.load(fptr, object_hook=json_ndarray_decode)

################################################################################

class NumpyEncoder(json.JSONEncoder):
    """
    extend JSONEncoder to handle numpy ndarray's
    https://stackoverflow.com/questions/26646362/nump-array-is-not-json-serializable
    """
    def default(self, o): # pylint: disable=E0202
        """method called by json.dump, when cls=NumpyEncoder"""
        if isinstance(o, np.ndarray):
            return {'__ndarray__':o.tolist()}
        return json.JSONEncoder.default(self, o)

def json_ndarray_decode(dct):
    """decode __ndarray__ tagged entries"""
    if '__ndarray__' in dct:
        return np.asarray(dct['__ndarray__'])
    return dct
