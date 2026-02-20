import argparse
import os
import sys

# Ensure repo root is on sys.path so demos can be run from the demos/ folder
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from atommover.utils.benchmarking import Benchmarking, BenchmarkingFigure
from atommover.utils.core import Configurations, PhysicalParams
from atommover.algorithms.single_species import PCFA, Hungarian, BalanceAndCompact, BCv2, ParallelLBAP, ParallelHungarian, GeneralizedBalance, Tetris
from atommover.utils.errormodels import ZeroNoise, UniformVacuumTweezerError, YbRydbergAODErrorModel


def main():
    parser = argparse.ArgumentParser(description="Run algorithm benchmarks (PCFA vs Hungarian)")
    parser.add_argument("--min_size", type=int, default=6, help="Minimum L for square target")
    parser.add_argument("--max_size", type=int, default=26, help="Maximum L for square target")
    parser.add_argument("--shots", type=int, default=20, help="Number of shots per configuration")
    parser.add_argument("--rounds", type=int, default=1, help="Rearrangement rounds")
    parser.add_argument("--save", action="store_true", help="Save xarray results to data/")
    parser.add_argument("--name", type=str, default=None, help="Save name (without extension)")
    args = parser.parse_args()

    # DONE: the number of rounds heavily influences the benchmarking results. 
    # For example, some algorithms may be very fast at preparing a configuration but have a low success rate,
    #  so they will perform well if only 1 round is allowed but will perform poorly if many rounds are allowed 
    # (since they will often fail and have to be rerun). On the other hand, some algorithms may be slower but more reliable, 
    # so they will perform better when more rounds are allowed. It would be interesting to benchmark how the performance 
    # of different algorithms changes as the number of rounds is increased. 
    # This could help users choose the best algorithm for their needs based on their tolerance for failure and their time constraints.
    # DONE: create production ready pipeline from end to end with minimal imaging with zero noise model to apply to real world experiments
    # TODO: Rework Presentation
    # DONE: include new plots   
    # DONE: find minimum amount of rounds and estimate with a formula how many rounds are needed for given error sources from Ybrydberg model.

    algos = [PCFA(), BCv2(), Hungarian(), BalanceAndCompact(), ParallelLBAP(), ParallelHungarian(), GeneralizedBalance(), Tetris()]
    targets = [Configurations.MIDDLE_FILL]
    params = [PhysicalParams()]
    sizes = list(range(args.min_size, args.max_size + 1, 4))

    if args.name is None:
        args.name = f"benchmark_{'_'.join(algo.__class__.__name__ for algo in algos)}"

    fig = BenchmarkingFigure(
        variables=["Wall time", "Mean moves", "Parallel move batches", "Success rate"],
        figure_type="scale",
    )
    bench = Benchmarking(
        algos=algos,
        target_configs=targets,
        error_models_list=[ZeroNoise(), UniformVacuumTweezerError(), YbRydbergAODErrorModel()],
        phys_params_list=params,
        sys_sizes=sizes,
        rounds_list=[args.rounds],
        n_shots=args.shots,
        n_species=1,
        figure_output=fig,
        per_round_logging=True,
        check_sufficient_atoms=True,
        show_progress=True
    )

    bench.run(do_ejection=False)
    if args.save:
        bench.save(args.name)

    # Optionally, plot results: comment in if needed
    bench.plot_results(save=True, savename=f"{args.name}_plot")

if __name__ == "__main__":
    main()
