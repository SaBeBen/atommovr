#!/usr/bin/env python3
"""Diagnostic script to run minimal benchmarks and catch errors."""

import sys
import traceback
from atommovr.utils.benchmarking import Benchmarking, BenchmarkingFigure
from atommovr.utils.core import Configurations, PhysicalParams
from atommovr.algorithms.single_species import PCFA, BalanceAndCompact
from atommovr.utils.errormodels import ZeroNoise

algos = [PCFA(), BalanceAndCompact()]
targets = [Configurations.MIDDLE_FILL]
error_models = [ZeroNoise()]
params = [PhysicalParams()]
sizes = [6]

fig = BenchmarkingFigure(
    variables=["Time", "Mean moves", "Parallel move batches", "Success rate"],
    figure_type="scale",
)

try:
    bench = Benchmarking(
        algos=algos,
        target_configs=targets,
        error_models_list=error_models,
        phys_params_list=params,
        sys_sizes=sizes,
        rounds_list=[1],
        n_shots=1,
        n_species=1,
        figure_output=fig,
        per_round_logging=False,
        check_sufficient_atoms=True,
        show_progress=True
    )
    print("Starting benchmark run...")
    bench.run(do_ejection=False)
    print("Benchmark completed successfully!")
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
