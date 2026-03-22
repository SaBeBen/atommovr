#!/usr/bin/env python3
"""Quick test of benchmark imports and initialization."""

from atommovr.utils.benchmarking import Benchmarking
from atommovr.utils.core import Configurations, PhysicalParams
from atommovr.algorithms.single_species import PCFA, BalanceAndCompact

print("Imports successful")

# Try creating a simple benchmark
algos = [PCFA(), BalanceAndCompact()]
targets = [Configurations.MIDDLE_FILL]
params = [PhysicalParams()]
sizes = [6]

try:
    bench = Benchmarking(
        algos=algos,
        target_configs=targets,
        error_models_list=[],
        phys_params_list=params,
        sys_sizes=sizes,
        rounds_list=[1],
        n_shots=1,
        n_species=1,
        figure_output=None,
        per_round_logging=False,
        check_sufficient_atoms=True,
        show_progress=False
    )
    print("Benchmark created successfully")
except Exception as e:
    print(f"Error creating benchmark: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
