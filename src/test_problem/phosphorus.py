"""phosphorus subclass of test_problem's TracerModuleState"""

import logging

from .tracer_module_state import TracerModuleState


class Phosphorus(TracerModuleState):
    """phosphorus tracer module specifics for TracerModuleState"""

    def __init__(self, tracer_module_name, fname):
        logger = logging.getLogger(__name__)
        logger.debug('iage, fname="%s"', fname)
        super().__init__(tracer_module_name, fname)
