import numpy as np
import pytest

from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.errormodels import ZeroNoise, UniformVacuumTweezerError

# Add new built-in models here as they are created
BUILTIN_MODELS = [
    ZeroNoise,
    UniformVacuumTweezerError,
]

REQUIRED_ATTRS = [
    "rng",
    "pickup_time",
    "putdown_time",
    "accel_time",
    "decel_time",
    "pickup_fail_rate",
    "putdown_fail_rate",
    "accel_fail_rate",
    "decel_fail_rate",
    "lifetime",
]

REQUIRED_METHODS = [
    "get_atom_loss",
    "apply_pickup_errors_mask",
    "apply_putdown_errors_mask",
    "apply_accel_errors_mask",
    "apply_decel_errors_mask",
    "apply_inevitable_collision_mask",
    "apply_avoidable_collision_mask",
]


@pytest.mark.parametrize("model_cls", BUILTIN_MODELS)
def test_builtin_models_are_errormodel_subclasses(model_cls) -> None:
    model = model_cls(seed=0)
    assert isinstance(model, ErrorModel)


@pytest.mark.parametrize("model_cls", BUILTIN_MODELS)
@pytest.mark.parametrize("attr_name", REQUIRED_ATTRS)
def test_builtin_models_have_required_attributes(model_cls, attr_name: str) -> None:
    model = model_cls(seed=0)
    assert hasattr(
        model, attr_name
    ), f"{model_cls.__name__} missing attribute {attr_name}"


@pytest.mark.parametrize("model_cls", BUILTIN_MODELS)
@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_builtin_models_have_required_methods(model_cls, method_name: str) -> None:
    model = model_cls(seed=0)
    method = getattr(model, method_name, None)
    assert callable(
        method
    ), f"{model_cls.__name__}.{method_name} is missing or not callable"


@pytest.mark.parametrize("model_cls", BUILTIN_MODELS)
def test_builtin_models_mask_methods_smoke(model_cls) -> None:
    model = model_cls(seed=0)
    event_mask = np.zeros(8, dtype=np.uint64)
    eligible = np.array(
        [True, False, True, False, True, False, True, False], dtype=bool
    )

    model.apply_pickup_errors_mask(event_mask, eligible)
    model.apply_putdown_errors_mask(event_mask, eligible)
    model.apply_accel_errors_mask(event_mask, eligible)
    model.apply_decel_errors_mask(event_mask, eligible)
    model.apply_inevitable_collision_mask(event_mask, eligible)
    model.apply_avoidable_collision_mask(event_mask, eligible)

    assert event_mask.shape == (8,)
    assert event_mask.dtype == np.uint64


@pytest.mark.parametrize("model_cls", BUILTIN_MODELS)
def test_builtin_models_get_atom_loss_contract_single_species(model_cls) -> None:
    model = model_cls(seed=0)
    state = np.ones((4, 4), dtype=np.uint8)

    new_state, loss_flag = model.get_atom_loss(state, evolution_time=0.1, n_species=1)

    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == state.shape
    assert isinstance(loss_flag, bool)
