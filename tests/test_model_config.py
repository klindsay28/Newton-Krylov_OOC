"""test functions in model_config.py"""

from src.model_config import (
    get_model_config_attr,
    get_precond_matrix_def,
    propagate_base_matrix_defs_to_all,
)

from .share import config_test_problem


def test_model_config():
    """create ModelConfig object and confirm that _model_config_obj is created"""

    config = config_test_problem()

    # confirm that _model_config_obj is created by getting an attr from it
    assert get_model_config_attr("modelinfo") == config["modelinfo"]


def test_propagate_base_matrix_defs_to_all():
    """test propagate_base_matrix_defs_to_all"""

    config_test_problem()

    base_def = get_precond_matrix_def("base")
    phosphorus = get_precond_matrix_def("phosphorus")

    # verify that base_def's hist_to_precond_varnames were propagated to phosphorus
    for varname in base_def["hist_to_precond_varnames"]:
        assert varname in phosphorus["hist_to_precond_varnames"]

    # add a hist var to base, re-propagate, and verify that it is added to phosphorus
    base_def["hist_to_precond_varnames"].append("new_hist_var")
    propagate_base_matrix_defs_to_all(get_model_config_attr("precond_matrix_defs"))
    assert "new_hist_var" in phosphorus["hist_to_precond_varnames"]

    # add a matrix opt to base, re-propagate, and verify that it is added to phosphorus
    base_def["precond_matrices_opts"] = ["matrix_opt_A sub_opt"]
    propagate_base_matrix_defs_to_all(get_model_config_attr("precond_matrix_defs"))
    assert "precond_matrices_opts" in phosphorus
    assert "matrix_opt_A sub_opt" in phosphorus["precond_matrices_opts"]

    # add a matrix opt base and phosphorus matrices, with different sub_opts,
    # re-propagate, and verify that phosphorus sub_opt doesn't change
    # verify that matrix_opt_A wasn't added again
    base_def["precond_matrices_opts"].append("matrix_opt_B sub_opt_base")
    phosphorus["precond_matrices_opts"].append("matrix_opt_B sub_opt_phosphorus")
    propagate_base_matrix_defs_to_all(get_model_config_attr("precond_matrix_defs"))
    assert "precond_matrices_opts" in phosphorus
    assert "matrix_opt_B sub_opt_phosphorus" in phosphorus["precond_matrices_opts"]
    assert "matrix_opt_B sub_opt_base" not in phosphorus["precond_matrices_opts"]
    assert phosphorus["precond_matrices_opts"].count("matrix_opt_A sub_opt") == 1
