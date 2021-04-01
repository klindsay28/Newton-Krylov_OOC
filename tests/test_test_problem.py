"""test test_problem features"""

import copy

from nk_ooc.test_problem.model_state import ModelState

from .share import config_test_problem


def test_depth_shared():
    """confirm that depth axis is shared, even in deep copies"""

    ModelState.model_config_obj = config_test_problem()

    model_state_a = ModelState("gen_init_iterate")
    assert model_state_a.tracer_modules[0].depth is model_state_a.depth

    model_state_b = ModelState("gen_init_iterate")
    assert model_state_a.depth is model_state_b.depth

    model_state_c = copy.deepcopy(model_state_b)
    assert model_state_c.depth is model_state_b.depth
    assert model_state_c.tracer_modules is not model_state_b.tracer_modules
