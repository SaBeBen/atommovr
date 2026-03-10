import numpy as np
import pytest

from atommover.utils.errormodels import ZeroNoise, UniformVacuumTweezerError
from atommover.utils.failure_policy import FailureBit, bit_value


# -------------------------
# ZeroNoise
# -------------------------
class TestZeroNoise:
    def test_repr(self) -> None:
        em = ZeroNoise()
        assert repr(em) == "ZeroNoise"


    def test_get_atom_loss_is_noop_single_species(self) -> None:
        em = ZeroNoise(seed=0)
        state = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        new_state, loss_flag = em.get_atom_loss(state, evolution_time=123.0, n_species=1)

        assert np.array_equal(new_state, state)
        assert loss_flag is False
        assert new_state is not state  # should return a copy


    def test_get_atom_loss_is_noop_dual_species(self) -> None:
        em = ZeroNoise(seed=0)
        state = np.ones((3, 4, 2), dtype=np.uint8)

        new_state, loss_flag = em.get_atom_loss(state, evolution_time=10.0, n_species=2)

        assert np.array_equal(new_state, state)
        assert loss_flag is False
        assert new_state is not state


    @pytest.mark.parametrize(
        "method_name",
        [
            "apply_pickup_errors_mask",
            "apply_putdown_errors_mask",
            "apply_accel_errors_mask",
            "apply_decel_errors_mask",
        ],
    )
    def test_mask_methods_do_nothing(self, method_name: str) -> None:
        em = ZeroNoise(seed=0)
        event_mask = np.zeros(8, dtype=np.uint64)
        eligible = np.ones(8, dtype=bool)

        getattr(em, method_name)(event_mask, eligible)

        assert np.all(event_mask == 0)


# -------------------------
# UniformVacuumTweezerError
# -------------------------
class TestUniformVacuumTweezerError:
    def test_repr(self) -> None:
        em = UniformVacuumTweezerError()
        assert repr(em) == "UniformVacuumTweezerError"


    def test_get_atom_loss_single_species_is_deterministic_for_seed(self) -> None:
        state = np.ones((20, 20), dtype=np.uint8)

        em1 = UniformVacuumTweezerError(lifetime=5.0, seed=123)
        em2 = UniformVacuumTweezerError(lifetime=5.0, seed=123)

        new1, flag1 = em1.get_atom_loss(state, evolution_time=0.7, n_species=1)
        new2, flag2 = em2.get_atom_loss(state, evolution_time=0.7, n_species=1)

        assert np.array_equal(new1, new2)
        assert flag1 == flag2


    def test_get_atom_loss_dual_species_is_deterministic_for_seed(self) -> None:
        state = np.ones((20, 20, 2), dtype=np.uint8)

        em1 = UniformVacuumTweezerError(lifetime=5.0, seed=456)
        em2 = UniformVacuumTweezerError(lifetime=5.0, seed=456)

        new1, flag1 = em1.get_atom_loss(state, evolution_time=0.7, n_species=2)
        new2, flag2 = em2.get_atom_loss(state, evolution_time=0.7, n_species=2)

        assert np.array_equal(new1, new2)
        assert flag1 == flag2


    def test_dual_species_applies_same_site_mask_to_both_channels(self) -> None:
        state = np.ones((12, 13, 2), dtype=np.uint8)
        em = UniformVacuumTweezerError(lifetime=3.0, seed=0)

        new_state, loss_flag = em.get_atom_loss(state, evolution_time=1.0, n_species=2)

        assert new_state.shape == state.shape
        assert np.array_equal(new_state[:, :, 0], new_state[:, :, 1])
        assert isinstance(loss_flag, bool)


    def test_invalid_n_species_raises(self) -> None:
        em = UniformVacuumTweezerError(seed=0)
        state = np.ones((4, 4), dtype=np.uint8)

        with pytest.raises(ValueError):
            em.get_atom_loss(state, evolution_time=1.0, n_species=3)


    def test_lifetime_nonpositive_raises_via_core(self) -> None:
        em = UniformVacuumTweezerError(lifetime=0.0, seed=0)
        state = np.ones((4, 4), dtype=np.uint8)

        with pytest.raises(ValueError):
            em.get_atom_loss(state, evolution_time=1.0, n_species=1)


    def test_REGRESSION_inherited_mask_method_uses_seeded_rng_and_sets_bits(self) -> None:
        """
        Regression test for child classes calling ErrorModel.__init__ so self.rng exists.
        """
        n = 16
        eligible = np.zeros(n, dtype=bool)
        eligible[::2] = True

        em = UniformVacuumTweezerError(pickup_fail_rate=1.0, seed=0)
        event_mask = np.zeros(n, dtype=np.uint64)

        em.apply_pickup_errors_mask(event_mask, eligible)

        bv = bit_value(FailureBit.PICKUP_FAIL)
        assert np.all(event_mask[eligible] == bv)
        assert np.all(event_mask[~eligible] == 0)


    @pytest.mark.slow
    def test_single_species_survival_rate_is_reasonable_statistically(self) -> None:
        rows, cols = 250, 250
        evolution_time = 0.8
        lifetime = 4.0
        p_survive = float(np.exp(-evolution_time / lifetime))

        state = np.ones((rows, cols), dtype=np.uint8)
        em = UniformVacuumTweezerError(lifetime=lifetime, seed=0)

        new_state, _ = em.get_atom_loss(state, evolution_time=evolution_time, n_species=1)

        phat = new_state.mean()
        n = rows * cols
        sigma = np.sqrt(p_survive * (1 - p_survive) / n)
        assert abs(phat - p_survive) <= 5 * sigma


    @pytest.mark.slow
    def test_dual_species_survival_rate_is_reasonable_statistically(self) -> None:
        rows, cols = 250, 250
        evolution_time = 1.1
        lifetime = 5.0
        p_survive = float(np.exp(-evolution_time / lifetime))

        state = np.ones((rows, cols, 2), dtype=np.uint8)
        em = UniformVacuumTweezerError(lifetime=lifetime, seed=1)

        new_state, _ = em.get_atom_loss(state, evolution_time=evolution_time, n_species=2)

        # Same site mask should be applied to both species
        assert np.array_equal(new_state[:, :, 0], new_state[:, :, 1])

        phat = new_state[:, :, 0].mean()
        n = rows * cols
        sigma = np.sqrt(p_survive * (1 - p_survive) / n)
        assert abs(phat - p_survive) <= 5 * sigma