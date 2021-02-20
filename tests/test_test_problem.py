"""test test_problem features"""

from src.test_problem.model_state import ModelState

from .share import config_test_problem


def test_depth_shared():
    """confirm that depth axis is shared"""

    config_test_problem()

    model_state_a = ModelState("gen_init_iterate")
    assert model_state_a.tracer_modules[0].depth is model_state_a.depth

    model_state_b = ModelState("gen_init_iterate")
    assert model_state_a.depth is model_state_b.depth
