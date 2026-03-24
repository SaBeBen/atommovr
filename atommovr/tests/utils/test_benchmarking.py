# Tests for benchmarking utility functions and classes

import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock

from atommovr.utils.benchmarking import (
    evaluate_moves,
    BenchmarkingFigure,
    Benchmarking,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.Move import Move
from atommovr.utils.core import Configurations, PhysicalParams
from atommovr.utils.errormodels import ZeroNoise
from atommovr.algorithms.Algorithm_class import Algorithm

###########################
# Test evaluate_moves     #
###########################


class TestEvaluateMoves:
    """Tests for the evaluate_moves function."""

    def test_empty_move_list(self):
        """Test with empty move list returns zero time and counts."""
        array = AtomArray(shape=[3, 3])
        array.matrix = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8
        ).reshape(3, 3, 1)

        result_array, t_total, counts = evaluate_moves(array, [])

        assert t_total == 0.0
        assert counts == [0, 0]

    def test_single_move_set(self):
        """Test with a single move set."""
        array = AtomArray(shape=[3, 3])
        array.matrix = np.array(
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8
        ).reshape(3, 3, 1)

        move_list = [[Move(0, 0, 0, 1)]]
        result_array, t_total, counts = evaluate_moves(array, move_list)

        assert counts[0] == 1  # N_parallel_moves
        assert counts[1] == 1  # N_non_parallel_moves
        assert isinstance(t_total, float)

    def test_multiple_move_sets(self):
        """Test with multiple move sets."""
        array = AtomArray(shape=[4, 4])
        array.matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        ).reshape(4, 4, 1)

        move_list = [
            [Move(0, 0, 0, 1)],
            [Move(1, 1, 1, 2)],
        ]
        result_array, t_total, counts = evaluate_moves(array, move_list)

        assert counts[0] == 2  # N_parallel_moves
        assert counts[1] == 2  # N_non_parallel_moves (1 + 1)

    def test_parallel_moves_in_single_set(self):
        """Test counting parallel moves within a single set."""
        array = AtomArray(shape=[4, 4])
        array.matrix = np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        ).reshape(4, 4, 1)

        # Two moves that can be executed in parallel
        move_list = [[Move(0, 0, 1, 0), Move(0, 3, 1, 3)]]
        result_array, t_total, counts = evaluate_moves(array, move_list)

        assert counts[0] == 1  # N_parallel_moves (one parallel set)
        assert counts[1] == 2  # N_non_parallel_moves (two individual moves)

    def test_returns_updated_array(self):
        """Test that the function returns the updated array."""
        array = AtomArray(shape=[2, 2])
        array.matrix = np.array([[1, 0], [0, 0]], dtype=np.uint8).reshape(2, 2, 1)

        move_list = [[Move(0, 0, 0, 1)]]
        result_array, _, _ = evaluate_moves(array, move_list)

        assert result_array is array  # Same object
        assert result_array.matrix[0, 1, 0] == 1
        assert result_array.matrix[0, 0, 0] == 0

    def test_time_accumulates(self):
        """Test that time accumulates across move sets."""
        array = AtomArray(shape=[3, 3])
        array.matrix = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8
        ).reshape(3, 3, 1)

        move_list = [
            [Move(0, 0, 0, 1)],
            [Move(1, 1, 1, 2)],
        ]
        _, t_total, _ = evaluate_moves(array, move_list)

        assert t_total >= 0.0


###########################
# Test BenchmarkingFigure #
###########################


class TestBenchmarkingFigureInit:
    """Tests for BenchmarkingFigure initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        fig = BenchmarkingFigure()
        assert fig.y_axis_variables == ["Success rate"]
        assert fig.figure_type == "scale"

    def test_custom_variables(self):
        """Test initialization with custom variables."""
        fig = BenchmarkingFigure(variables=["Success rate", "Time"])
        assert fig.y_axis_variables == ["Success rate", "Time"]

    def test_all_valid_variables(self):
        """Test initialization with all valid variables."""
        all_vars = [
            "Success rate",
            "Filling fraction",
            "Time",
            "Wrong places #",
            "Total atoms",
        ]
        fig = BenchmarkingFigure(variables=all_vars)
        assert fig.y_axis_variables == all_vars

    def test_custom_figure_type_hist(self):
        """Test initialization with histogram figure type."""
        fig = BenchmarkingFigure(figure_type="hist")
        assert fig.figure_type == "hist"

    def test_custom_figure_type_pattern(self):
        """Test initialization with pattern figure type."""
        fig = BenchmarkingFigure(figure_type="pattern")
        assert fig.figure_type == "pattern"

    def test_invalid_variable_raises_keyerror(self):
        """Test that invalid variable raises KeyError."""
        with pytest.raises(KeyError):
            BenchmarkingFigure(variables=["Invalid variable"])

    def test_mixed_valid_invalid_variables(self):
        """Test that mix of valid and invalid variables raises KeyError."""
        with pytest.raises(KeyError):
            BenchmarkingFigure(variables=["Success rate", "Invalid"])

    def test_empty_variables_list(self):
        """Test initialization with empty variables list."""
        fig = BenchmarkingFigure(variables=[])
        assert fig.y_axis_variables == []


class TestBenchmarkingFigureGenerateMethods:
    """Tests for BenchmarkingFigure figure generation methods."""

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_scaling_figure_no_save(self, mock_savefig, mock_subplots):
        """Test generate_scaling_figure without saving."""
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        fig = BenchmarkingFigure(variables=["Success rate"])
        x_axis = [10, 11, 12]
        results = [
            {"Success rate": 0.8, "algorithm": Algorithm()},
            {"Success rate": 0.85, "algorithm": Algorithm()},
            {"Success rate": 0.9, "algorithm": Algorithm()},
        ]

        fig.generate_scaling_figure(x_axis, results, "Test", "Size", save=False)

        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_scaling_figure_with_save(self, mock_savefig, mock_subplots):
        """Test generate_scaling_figure with saving."""
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        fig = BenchmarkingFigure(variables=["Success rate"])
        x_axis = [10, 11, 12]
        results = [
            {"Success rate": 0.8, "algorithm": Algorithm()},
            {"Success rate": 0.85, "algorithm": Algorithm()},
            {"Success rate": 0.9, "algorithm": Algorithm()},
        ]

        fig.generate_scaling_figure(
            x_axis, results, "Test", "Size", save=True, savename="test_fig"
        )

        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    def test_generate_scaling_figure_raises_on_nan(self, mock_subplots):
        """Test that NaN values in results raise exception."""
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        fig = BenchmarkingFigure(variables=["Success rate"])
        x_axis = [10]
        results = [{"Success rate": float("nan"), "algorithm": Algorithm()}]

        with pytest.raises(Exception, match="nan"):
            fig.generate_scaling_figure(x_axis, results, "Test", "Size", save=False)

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_histogram_figure_no_save(self, mock_savefig, mock_subplots):
        """Test generate_histogram_figure without saving."""
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        fig = BenchmarkingFigure(variables=["Success rate"])
        results = [{"Success rate": 0.8, "algorithm": Algorithm()}]

        fig.generate_histogram_figure(results, "Test", "Size", save=False)

        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_histogram_figure_with_save(self, mock_savefig, mock_subplots):
        """Test generate_histogram_figure with saving."""
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        fig = BenchmarkingFigure(variables=["Success rate"])
        results = [{"Success rate": 0.8, "algorithm": Algorithm()}]

        fig.generate_histogram_figure(
            results, "Test", "Size", save=True, savename="test_hist"
        )

        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_pattern_figure_no_save(self, mock_savefig, mock_subplots):
        """Test generate_pattern_figure without saving."""
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        fig = BenchmarkingFigure(variables=["Success rate"])
        x_axis = [10, 11, 12]
        results = [
            {"Success rate": 0.8, "target": Configurations.MIDDLE_FILL},
            {"Success rate": 0.85, "target": Configurations.MIDDLE_FILL},
            {"Success rate": 0.9, "target": Configurations.MIDDLE_FILL},
        ]

        fig.generate_pattern_figure(x_axis, results, "Test", "Size", save=False)

        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.subplots")
    def test_generate_scaling_figure_list_variable_averaged(self, mock_subplots):
        """Test that list variables are averaged."""
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        fig = BenchmarkingFigure(variables=["Filling fraction"])
        x_axis = [10]
        results = [{"Filling fraction": [0.7, 0.8, 0.9], "algorithm": Algorithm()}]

        # Should not raise, list gets averaged
        fig.generate_scaling_figure(x_axis, results, "Test", "Size", save=False)


###########################
# Test Benchmarking Class #
###########################


class TestBenchmarkingInit:
    """Tests for Benchmarking class initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        bench = Benchmarking()

        assert bench.n_algos == 1
        assert bench.n_shots == 100
        assert bench.check_sufficient_atoms  # == True
        assert bench.istargetlist  # == True

    def test_custom_algorithms(self):
        """Test initialization with custom algorithms."""
        algos = [Algorithm(), Algorithm()]
        bench = Benchmarking(algos=algos)

        assert bench.n_algos == 2
        assert len(bench.algos) == 2

    def test_custom_system_sizes(self):
        """Test initialization with custom system sizes."""
        sizes = [4, 5, 6]
        bench = Benchmarking(sys_sizes=sizes)

        assert bench.n_sizes == 3
        assert bench.system_size_range == sizes

    def test_custom_n_shots(self):
        """Test initialization with custom n_shots."""
        bench = Benchmarking(n_shots=50)
        assert bench.n_shots == 50

    def test_custom_n_species(self):
        """Test initialization with custom n_species."""
        bench = Benchmarking(n_species=2)
        assert bench.tweezer_array.n_species == 2

    def test_check_sufficient_atoms_false(self):
        """Test initialization with check_sufficient_atoms=False."""
        bench = Benchmarking(check_sufficient_atoms=False)
        assert not bench.check_sufficient_atoms  # == False

    def test_target_configs_as_list(self):
        """Test initialization with target configs as list."""
        targets = [Configurations.MIDDLE_FILL, Configurations.CHECKERBOARD]
        bench = Benchmarking(target_configs=targets)

        assert bench.istargetlist  # == True
        assert bench.n_targets == 2

    def test_target_configs_as_ndarray(self):
        """Test initialization with target configs as ndarray."""
        # Create explicit target configs for each system size
        sys_sizes = [4, 5]
        targets = np.array(
            [
                [np.ones((4, 4)), np.zeros((4, 4))],
                [np.ones((5, 5)), np.zeros((5, 5))],
            ],
            dtype=object,
        )

        bench = Benchmarking(target_configs=targets, sys_sizes=sys_sizes)

        assert not bench.istargetlist  # == False
        assert bench.n_targets == 2

    def test_target_configs_ndarray_wrong_shape_raises(self):
        """Test that ndarray target configs with wrong shape raises IndexError."""
        sys_sizes = [4, 5, 6]  # 3 sizes
        targets = np.array(
            [
                [np.ones((4, 4))],
                [np.ones((5, 5))],
            ],
            dtype=object,
        )  # Only 2 sizes

        with pytest.raises(IndexError):
            Benchmarking(target_configs=targets, sys_sizes=sys_sizes)

    def test_target_configs_invalid_type_raises(self):
        """Test that invalid target config type raises TypeError."""
        with pytest.raises(TypeError):
            Benchmarking(target_configs="invalid")

    def test_custom_error_models(self):
        """Test initialization with custom error models."""
        models = [ZeroNoise(), ZeroNoise()]
        bench = Benchmarking(error_models_list=models)

        assert bench.n_models == 2

    def test_custom_phys_params(self):
        """Test initialization with custom physical params."""
        params = [PhysicalParams(), PhysicalParams(AOD_speed=0.2)]
        bench = Benchmarking(phys_params_list=params)

        assert bench.n_parsets == 2

    def test_custom_rounds_list(self):
        """Test initialization with custom rounds list."""
        rounds = [1, 2, 3]
        bench = Benchmarking(rounds_list=rounds)

        assert bench.n_rounds == 3
        assert bench.rounds_list == rounds


class TestBenchmarkingSetObservables:
    """Tests for Benchmarking.set_observables method."""

    def test_set_single_observable(self):
        """Test setting a single observable."""
        bench = Benchmarking()
        bench.set_observables(["Time"])

        assert bench.figure_output.y_axis_variables == ["Time"]

    def test_set_multiple_observables(self):
        """Test setting multiple observables."""
        bench = Benchmarking()
        observables = ["Success rate", "Time", "Filling fraction"]
        bench.set_observables(observables)

        assert bench.figure_output.y_axis_variables == observables

    def test_set_empty_observables(self):
        """Test setting empty observables list."""
        bench = Benchmarking()
        bench.set_observables([])

        assert bench.figure_output.y_axis_variables == []


class TestBenchmarkingGetResultArrayDims:
    """Tests for Benchmarking.get_result_array_dims method."""

    def test_updates_dimensions_from_list_targets(self):
        """Test that dimensions are correctly updated for list targets."""
        bench = Benchmarking(
            algos=[Algorithm(), Algorithm()],
            target_configs=[Configurations.MIDDLE_FILL, Configurations.CHECKERBOARD],
            sys_sizes=[4, 5, 6],
        )

        bench.get_result_array_dims()

        assert bench.n_algos == 2
        assert bench.n_targets == 2
        assert bench.n_sizes == 3

    def test_updates_dimensions_after_modification(self):
        """Test that dimensions update after modifying attributes."""
        bench = Benchmarking()

        # Modify attributes
        bench.algos = [Algorithm(), Algorithm(), Algorithm()]
        bench.system_size_range = [10, 11]

        bench.get_result_array_dims()

        assert bench.n_algos == 3
        assert bench.n_sizes == 2


class TestBenchmarkingRunBenchmarkRound:
    """Tests for Benchmarking._run_benchmark_round method."""

    def test_invalid_num_rounds_zero_raises(self):
        """Test that num_rounds=0 raises ValueError."""
        bench = Benchmarking(sys_sizes=[4], n_shots=1)
        bench.tweezer_array.shape = [4, 4]
        bench.tweezer_array.generate_target(Configurations.MIDDLE_FILL)
        bench.init_config_storage = [np.ones((4, 4))]

        with pytest.raises(ValueError, match="cannot be 0"):
            bench._run_benchmark_round(Algorithm(), num_rounds=0)

    def test_invalid_num_rounds_negative_raises(self):
        """Test that negative num_rounds raises ValueError."""
        bench = Benchmarking(sys_sizes=[4], n_shots=1)
        bench.tweezer_array.shape = [4, 4]
        bench.tweezer_array.generate_target(Configurations.MIDDLE_FILL)
        bench.init_config_storage = [np.ones((4, 4))]

        with pytest.raises(ValueError, match="cannot be 0, negative"):
            bench._run_benchmark_round(Algorithm(), num_rounds=-1)

    def test_invalid_num_rounds_float_raises(self):
        """Test that float num_rounds raises ValueError."""
        bench = Benchmarking(sys_sizes=[4], n_shots=1)
        bench.tweezer_array.shape = [4, 4]
        bench.tweezer_array.generate_target(Configurations.MIDDLE_FILL)
        bench.init_config_storage = [np.ones((4, 4))]

        with pytest.raises(ValueError, match="non-integer"):
            bench._run_benchmark_round(Algorithm(), num_rounds=1.5)

    def test_returns_tuple_of_correct_length(self):
        """Test that method returns tuple with 7 elements."""
        bench = Benchmarking(sys_sizes=[4], n_shots=2)
        bench.tweezer_array.shape = [4, 4]
        bench.tweezer_array.generate_target(Configurations.MIDDLE_FILL)
        bench.init_config_storage = [np.ones((4, 4)) for _ in range(2)]

        result = bench._run_benchmark_round(
            Algorithm(), pattern=Configurations.MIDDLE_FILL
        )

        assert len(result) == 7

    def test_filling_fractions_are_list(self):
        """Test that filling_fractions is a list."""
        bench = Benchmarking(sys_sizes=[4], n_shots=3)
        bench.tweezer_array.shape = [4, 4]
        bench.tweezer_array.generate_target(Configurations.MIDDLE_FILL)
        bench.init_config_storage = [np.ones((4, 4)) for _ in range(3)]

        _, _, fill_fracs, _, _, _, _ = bench._run_benchmark_round(
            Algorithm(), pattern=Configurations.MIDDLE_FILL
        )

        assert isinstance(fill_fracs, list)
        assert len(fill_fracs) == 3

    def test_wrong_places_are_list(self):
        """Test that wrong_places is a list."""
        bench = Benchmarking(sys_sizes=[4], n_shots=3)
        bench.tweezer_array.shape = [4, 4]
        bench.tweezer_array.generate_target(Configurations.MIDDLE_FILL)
        bench.init_config_storage = [np.ones((4, 4)) for _ in range(3)]

        _, _, _, wrong_places, _, _, _ = bench._run_benchmark_round(
            Algorithm(), pattern=Configurations.MIDDLE_FILL
        )

        assert isinstance(wrong_places, list)
        assert len(wrong_places) == 3


class TestBenchmarkingRun:
    """Tests for Benchmarking.run method."""

    def test_run_creates_benchmarking_results(self):
        """Test that run() creates benchmarking_results attribute."""
        bench = Benchmarking(
            sys_sizes=[4],
            n_shots=2,
            target_configs=[Configurations.MIDDLE_FILL],
        )

        bench.run()

        assert hasattr(bench, "benchmarking_results")
        assert isinstance(bench.benchmarking_results, xr.Dataset)

    def test_run_results_have_expected_variables(self):
        """Test that benchmarking_results has expected data variables."""
        bench = Benchmarking(
            sys_sizes=[4],
            n_shots=2,
            target_configs=[Configurations.MIDDLE_FILL],
        )

        bench.run()

        expected_vars = [
            "success rate",
            "time",
            "filling fraction",
            "wrong places",
            "n atoms",
            "n targets",
            "sufficient rate",
        ]
        for var in expected_vars:
            assert var in bench.benchmarking_results

    def test_run_with_ejection(self):
        """Test run() with do_ejection=True."""
        bench = Benchmarking(
            sys_sizes=[4],
            n_shots=2,
            target_configs=[Configurations.MIDDLE_FILL],
        )

        bench.run(do_ejection=True)

        assert hasattr(bench, "benchmarking_results")


class TestBenchmarkingPlotResults:
    """Tests for Benchmarking.plot_results method."""

    @patch.object(BenchmarkingFigure, "generate_scaling_figure")
    def test_plot_results_scale_type(self, mock_generate):
        """Test plot_results with scale figure type."""
        bench = Benchmarking(
            sys_sizes=[4],
            n_shots=2,
            figure_output=BenchmarkingFigure(figure_type="scale"),
        )
        bench.run()

        bench.plot_results()

        mock_generate.assert_called_once()

    @patch.object(BenchmarkingFigure, "generate_histogram_figure")
    def test_plot_results_hist_type(self, mock_generate):
        """Test plot_results with histogram figure type."""
        bench = Benchmarking(
            sys_sizes=[4],
            n_shots=2,
            figure_output=BenchmarkingFigure(figure_type="hist"),
        )
        bench.run()

        bench.plot_results()

        mock_generate.assert_called_once()

    @patch.object(BenchmarkingFigure, "generate_pattern_figure")
    def test_plot_results_pattern_type(self, mock_generate):
        """Test plot_results with pattern figure type."""
        bench = Benchmarking(
            sys_sizes=[4],
            n_shots=2,
            figure_output=BenchmarkingFigure(figure_type="pattern"),
        )
        bench.run()

        bench.plot_results()

        mock_generate.assert_called_once()

    @patch.object(BenchmarkingFigure, "generate_scaling_figure")
    def test_plot_results_custom_savename(self, mock_generate):
        """Test plot_results with custom savename."""
        bench = Benchmarking(
            sys_sizes=[4],
            n_shots=2,
        )
        bench.run()

        bench.plot_results(save=True, savename="custom_name")

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args
        assert call_kwargs[1]["savename"] == "custom_name"


class TestBenchmarkingSaveLoad:
    """Tests for Benchmarking save/load methods."""

    def test_save_strips_nc_extension(self, tmp_path, monkeypatch):
        """Test that save() handles .nc extension correctly."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        bench = Benchmarking(sys_sizes=[4], n_shots=2)
        bench.run()

        # Save with .nc extension
        bench.save("test_results.nc")

        assert (tmp_path / "data" / "test_results.nc").exists()

    def test_save_adds_nc_extension(self, tmp_path, monkeypatch):
        """Test that save() adds .nc extension if missing."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        bench = Benchmarking(sys_sizes=[4], n_shots=2)
        bench.run()

        # Save without .nc extension
        bench.save("test_results")

        assert (tmp_path / "data" / "test_results.nc").exists()

    def test_load_strips_nc_extension(self, tmp_path, monkeypatch):
        """Test that load() handles .nc extension correctly."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        # Create and save data
        bench1 = Benchmarking(sys_sizes=[4], n_shots=2)
        bench1.run()
        bench1.save("test_load")

        # Load into new instance
        bench2 = Benchmarking()
        bench2.load("test_load.nc")

        assert hasattr(bench2, "benchmarking_results")

    def test_load_without_extension(self, tmp_path, monkeypatch):
        """Test that load() works without .nc extension."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        # Create and save data
        bench1 = Benchmarking(sys_sizes=[4], n_shots=2)
        bench1.run()
        bench1.save("test_load2")

        # Load into new instance
        bench2 = Benchmarking()
        bench2.load("test_load2")

        assert hasattr(bench2, "benchmarking_results")


class TestBenchmarkingLoadParamsFromDataset:
    """Tests for Benchmarking.load_params_from_dataset method."""

    def test_loads_algorithm_values(self):
        """Test that algorithm values are loaded from dataset."""
        # Create dataset
        bench1 = Benchmarking(
            algos=[Algorithm()],
            sys_sizes=[4],
            n_shots=2,
        )
        bench1.run()

        # Load params
        bench2 = Benchmarking()
        bench2.load_params_from_dataset(bench1.benchmarking_results)

        assert len(bench2.algos) == 1

    def test_loads_system_sizes(self):
        """Test that system sizes are loaded from dataset."""
        bench1 = Benchmarking(
            sys_sizes=[4, 5],
            n_shots=2,
        )
        bench1.run()

        bench2 = Benchmarking()
        bench2.load_params_from_dataset(bench1.benchmarking_results)

        assert list(bench2.system_size_range) == [4, 5]

    def test_loads_n_shots(self):
        """Test that n_shots is loaded from dataset."""
        bench1 = Benchmarking(
            sys_sizes=[4],
            n_shots=5,
        )
        bench1.run()

        bench2 = Benchmarking()
        bench2.load_params_from_dataset(bench1.benchmarking_results)

        assert bench2.n_shots == 5

    def test_loads_rounds_list_as_integers(self):
        """Test that rounds_list values are converted to integers."""
        bench1 = Benchmarking(
            sys_sizes=[4],
            n_shots=2,
            rounds_list=[1, 2],
        )
        bench1.run()

        bench2 = Benchmarking()
        bench2.load_params_from_dataset(bench1.benchmarking_results)

        assert bench2.rounds_list == [1, 2]
        assert all(isinstance(r, int) for r in bench2.rounds_list)


###########################
# Integration Tests       #
###########################


class TestBenchmarkingIntegration:
    """Integration tests for the Benchmarking class."""

    def test_full_benchmarking_workflow_single_species(self):
        """Test complete benchmarking workflow for single species."""
        bench = Benchmarking(
            algos=[Algorithm()],
            target_configs=[Configurations.MIDDLE_FILL],
            sys_sizes=[4],
            n_shots=3,
            n_species=1,
        )

        bench.run()

        # Verify results structure
        assert "success rate" in bench.benchmarking_results
        assert "time" in bench.benchmarking_results

        # Verify dimensions
        sr = bench.benchmarking_results["success rate"]
        assert sr.dims == (
            "algorithm",
            "target",
            "sys size",
            "error model",
            "physical params",
            "num rounds",
        )

    def test_full_benchmarking_workflow_dual_species(self):
        """Test complete benchmarking workflow for dual species."""
        bench = Benchmarking(
            algos=[Algorithm()],
            target_configs=[Configurations.CHECKERBOARD],
            sys_sizes=[4],
            n_shots=2,
            n_species=2,
        )

        bench.run()

        assert hasattr(bench, "benchmarking_results")

    def test_multiple_algorithms_comparison(self):
        """Test benchmarking with multiple algorithms."""
        bench = Benchmarking(
            algos=[Algorithm(), Algorithm()],
            target_configs=[Configurations.MIDDLE_FILL],
            sys_sizes=[4],
            n_shots=2,
        )

        bench.run()

        # Should have results for both algorithms
        assert bench.benchmarking_results["success rate"].shape[0] == 2

    def test_multiple_target_configs_comparison(self):
        """Test benchmarking with multiple target configurations."""
        bench = Benchmarking(
            algos=[Algorithm()],
            target_configs=[Configurations.MIDDLE_FILL, Configurations.CHECKERBOARD],
            sys_sizes=[4],
            n_shots=2,
        )

        bench.run()

        # Should have results for both targets
        assert bench.benchmarking_results["success rate"].shape[1] == 2

    def test_multiple_system_sizes(self):
        """Test benchmarking with multiple system sizes."""
        bench = Benchmarking(
            algos=[Algorithm()],
            target_configs=[Configurations.MIDDLE_FILL],
            sys_sizes=[4, 5],
            n_shots=2,
        )

        bench.run()

        # Should have results for both sizes
        assert bench.benchmarking_results["success rate"].shape[2] == 2
